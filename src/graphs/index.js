import { Annotation, messagesStateReducer, StateGraph, START, END, Command} from '@langchain/langgraph'
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod'
import {PromptTemplate} from '@langchain/core/prompts'
import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import { search } from '../tools/baidu'

const MAX_TURNS = 10;
let llm;

const ParentState = Annotation.Root({
	// --- 配置 ---
	modelConfig: Annotation({}),

	// --- 输入与历史 ---
	userInput: Annotation({}),
	messages: Annotation({
		reducer: messagesStateReducer,
		default: () => []
	}),

	// --- 记忆与上下文 ---
	memory: Annotation({}),
	summary: Annotation({}),
	tool_outputs: Annotation({}),

	// --- Planner的决策输出 ---
	// 这些字段由 plannerNode 生成，用于驱动下一步流程
	planner_thought: Annotation({}),
	nextAction: Annotation({}),
	tool_calls: Annotation({}),
	turn_count: Annotation({ }),
});

const entry = async (state) =>{
	console.log('Enter Entry Node');
	llm = new ChatOpenAI({
		modelName: state.modelConfig.model,
		temperature: state.modelConfig.temperature,
		streaming: true,
		apiKey: state.modelConfig.token,
		configuration: {
			baseURL: "https://ark.cn-beijing.volces.com/api/v3/",
		},
		modelKwargs: {
			disable_thinking: true,
		}
	})
	const lastMessage = state.messages[state.messages.length - 1];
	return {userInput: lastMessage?.content || '', turn_count: 0 };
}

const memoryRetrieverNode = async(state)=>{
	const retrievedMemories = [
		'坐标: 北京',
		'年龄: 4岁',
		'喜欢的活动: 露营，游乐场, 过山车等'
	]

	let memoryString;
	if (retrievedMemories.length > 0) {
		memoryString = "检索到以下历史偏好信息，请在本次规划中重点参考：\n- " + retrievedMemories.join('\n- ');
	} else {
		// 关键改动：返回一个明确的“空状态”消息
		memoryString = "【状态明确】：未检索到与当前用户相关的历史偏好信息。";
	}
	return { memory: memoryString };
}

const plannerNode = async(state) => {
	console.log(`Enter Planner Node, turn count: ${state.turn_count}` );
	const systemPrompt = new SystemMessage(`# 角色定义
你是一个名为“亲子旅行规划师”的AI助手。你的核心任务是与用户互动，规划一次完美的、个性化的带娃自驾旅行。
---
## 可用工具定义 (Available Tools Definition)

你拥有以下工具来帮助你完成规划。在生成 \`tool_calls\` 时，你必须严格使用这里定义的 \`tool_name\` 和 \`parameters\`。

### 1. 百度搜索 (Baidu Search)
- **tool_name**: \`baidu_search\`
- **description**: 一个通用的搜索引擎，可以用来查询最新的旅游攻略、目的地的具体信息（如停车情况、游玩项目、开放时间）等。
- **parameters**:
  - \`name\`: \`query\`
  - \`type\`: \`string\`
  - \`description\`: 你想要搜索的关键词或问题。例如："北京野生动物园 停车攻略" 或 "5岁孩子喜欢什么样的游乐项目"。
  - \`required\`: true

### 2. 和风天气 (QWeather)
- **tool_name**: \`qweather\`
- **description**: 查询指定地点在未来某一天的天气情况，包括逐小时预报。
- **parameters**:
  - \`name\`: \`location\`
  - \`type\`: \`string\`
  - \`description\`: 需要查询天气的具体地理位置，例如："北京野生动物园" 或 "北京市大兴区"。
  - \`required\`: true
  - \`name\`: \`date\`
  - \`type\`: \`string\`
  - \`description\`: 需要查询的日期，格式为 "YYYY-MM-DD"。例如："2025-06-14"。
  - \`required\`: true

### 3. 高德地图 (Gaode Maps)
- **tool_name**: \`gaode_maps\`
- **description**: 用于查询车辆导航信息，包括预计时长、实时路况和路线规划。
- **parameters**:
  - \`name\`: \`start\`
  - \`type\`: \`string\`
  - \`description\`: 导航的起点位置。例如："北京市海淀区中关村"。
  - \`required\`: true
  - \`name\`: \`end\`
  - \`type\`: \`string\`
  - \`description\`: 导航的终点位置。例如："北京野生动物园"。
  - \`required\`: true

## 工具辅助信息

### 当前时间戳
如果需要计算相对时间，请使用当前时间戳作为基准：\`${new Date()}\`。

---

# 核心原则
1.  **节约至上 (API Efficiency First)**: 你必须严格控制API工具的调用。绝不能一次性调用所有工具查询所有信息。你的目标是“按需查询”，只在绝对必要或用户明确要求时才调用工具。
2.  **分步规划 (Phased Planning)**: 你的工作流程是分阶段的。首先是“初步建议”阶段，然后是“深化细节”阶段。严禁在“初步建议”阶段调用高成本的、针对特定地点的API（如天气、路况、停车详情）。
3.  **用户中心 (User-Centric)**: 始终将用户的选择和反馈作为推进规划的唯一依据。在给出初步建议后，必须等待用户选出1-2个候选方案，才能进入下一步。
4.  **记忆驱动 (Memory-Driven)**: 在规划开始时，你必须首先查询用户的历史偏好和禁忌，并将这些信息作为你提出建议的最高优先级依据。
5.  **解读明确的空状态 (Interpret Explicit Nulls)**: 在【历史偏好】区域，你可能会看到一条明确的“未检索到”或“状态明确”的消息。这表示上一步的检索已经执行过且结果为空，你**绝不能**因此再次调用\`memory_retriever\`工具。你应当基于其他可用信息继续规划。

# 工作流程 (Workflow)

你必须严格遵循以下思考和行动步骤：

**阶段一：初步建议与筛选 (Initial Suggestion & Filtering)**

1.  **启动与记忆检索**:
    * 当收到用户请求时，第一步是调用 \`memory_retriever\` 工具，查询用户的历史偏好（喜欢的活动、去过的地方）和明确的禁忌（不喜欢的、过敏等）。
    * **思考**: “根据用户的历史数据，他/她可能喜欢什么类型的目的地？有什么地方是绝对要排除的？”

2.  **广泛搜索与构思**:
    * **仅可使用** \`baidu_search\` 工具进行广泛和概括性的搜索。例如，搜索“北京周边适合5岁孩子的自然风景区”或“广州附近亲子农场推荐”。
    * **禁止**: 在此阶段使用\`gaode_maps\`或\`qweather\`。
    * **思考**: “结合用户的历史偏好和本次搜索结果，哪些地点看起来最合适？我要生成一个包含3-5个选项的列表。”

3.  **生成初步建议列表**:
    * 整合记忆和搜索结果，向用户提出一个包含3-5个候选目的地的列表。
    * 对于每个目的地，只提供**简要信息**，例如：
        * 名称
        * 一句话简介（如：“一个大型野生动物园，可以和动物近距离接触”）
        * 大致方位和预估车程（用模糊词语，如“约1.5小时车程”，而非精确导航结果）
    * **行动**: 以清晰的列表形式呈现给用户，并明确要求用户做出选择：“**以上是为您初步筛选的几个地方，您对哪个或者哪几个比较感兴趣？请告诉我，我再为您深入查询详细信息。**”

**阶段二：深化细节与最终方案 (Detailed Planning & Final Itinerary)**

1.  **等待用户选择**:
    * **绝对静默**: 在用户做出选择前，不进行任何新的工具调用。
    * **思考**: “用户选择了A和B，现在我需要为这两个地方收集详细信息了。”

2.  **按需调用详细信息工具**:
    * 针对用户选定的**一到两个**候选方案，依次调用高成本API：
        * 调用 \`baidu_search\` 查询具体的“**停车攻略**”和“**必玩项目推荐**”。
        * 调用 \`qweather\` 查询目的地的“**计划出游日的逐小时天气**”。
        * 调用 \`gaode_maps\` 查询从用户家（或默认出发点）到目的地的“**去程和回程的实时路况和精确时间**”。
    * **思考**: “我已经拿到了A地的天气、路况、停车和游玩攻略。信息是否完整？现在开始查询B地的信息。”

3.  **整合与决策支持**:
    * 将所有查询到的详细信息进行结构化整合。
    * 为用户提供一个清晰的对比，或一个完整的单点旅行计划。例如：“A地天气晴朗，但下午可能有风，路程约1.5小时，停车位较为充足... B地...”
    * **行动**: 向用户呈现最终方案，并询问是否确认。“**根据详细信息，A方案看起来更合适。这是为您规划好的具体行程，您看可以吗？**”

4.  **记录反馈**:
    * 在对话结束后，将本次交互中用户的选择（“最终选了动物园”）、和未明说的偏好（“选择了车程在2小时内的地点”）传递给 \`memory_recorder\` 工具进行存储，用于未来的规划。

# 总结
你的核心价值在于**智能地、节约地**调度工具，而不是一个简单的信息聚合器。始终记住：**先概括，再深入；先询问，再行动。**
	`)

	const ToolCallSchema = z.object({
		toolName: z.string().describe("要调用的工具名称，例如 'baidu_search' 或 'qweather'。"),
		toolArgs: z.record(z.string(), z.any()).describe("调用工具所需的参数，以键值对形式表示。"),
	});

	const PlannerOutputSchema = z.object({
		/** * Agent的内心独白和思考过程。
		 * 这部分内容会优先返回并展示给用户，让他知道你正在做什么，提升透明度和体验。
		 * 例如："用户想找个适合孩子的自然风光地，我需要先查询他的历史偏好，然后用搜索引擎找一些候选地点。"
		 */
		thought: z.string().describe(
			"你的思考过程或内心独白。这部分内容会展示给用户，让他知道你正在思考什么。"
		),

		/**
		 * 决定下一步流程走向的关键指令。
		 * 'tool_call': 表示需要调用一个工具来获取信息。
		 * 'ask_user': 表示需要向用户提问、呈现选项或等待反馈。
		 * 'finish': 表示所有任务已完成，可以生成最终答复并结束流程。
		 */
		nextAction: z.enum(['tool_call', 'ask_user', 'finish']).describe(
			"你决定下一步要执行的核心动作。"
		),

		tool_calls: z.array(ToolCallSchema).nullable().describe(
			"一个包含零个或多个工具调用请求的列表。当你需要同时获取多方面信息时，请在此列表中提供所有必要的工具调用。"
		),

		/**
		 * 准备呈现给用户的、经过润色的友好回复。
		 * 仅在 nextAction 是 'ask_user' 或 'finish' 时使用。
		 * 当你需要向用户呈现选项列表或最终方案时，内容会在这里。
		 */
		responseToUser: z.string().nullable().describe(
			"当 nextAction 是 'ask_user' 时，这里是你向用户提问的**简洁问题**。当 nextAction 是" +
			" 'finish' 时，**此字段可留空或只包含一个简单的确认，因为将有另一个节点负责生成最终的详细回复**。"
		),
	});

	let humanMessageContent = '';

	if(state.summary) {
		humanMessageContent += `### 对话摘要 (Conversation Summary)\n${state.summary}\n\n`;
	}

	if (state.memory) {
		humanMessageContent += `### 历史偏好 (Retrieved Memories)\n${state.memory}\n\n`;
	}

	// **关键**: 只有当 tool_outputs 存在时才添加这部分
	if (state.tool_outputs && state.tool_outputs.length > 0) {
		const formattedOutputs = state.tool_outputs
			.map(r => `工具 "${r.toolName}" 的返回结果:\n${r.output}`)
			.join('\n\n');
		humanMessageContent += `### 最近的工具执行结果 (Latest Tool Outputs)\n${formattedOutputs}\n\n`;
	}

	// 添加最近的对话历史 (例如最后4条)
	const recentHistory = state.messages.slice(-4)
		.map(msg => `${msg.role}: ${msg.content}`).join('\n');
	if (recentHistory) {
		humanMessageContent += `### 最近的对话历史 (Recent Conversation History)\n${recentHistory}\n\n`;
	}

	humanMessageContent += `### 当前用户的最新请求 (Current User Request)\n${state.userInput}\n\n`;

	humanMessageContent += `### 你的任务 (Your Task)\n基于以上所有信息，进行思考，并决定下一步行动。你的输出必须严格遵守预定义的JSON Schema。`;


	const response = await llm.withStructuredOutput(PlannerOutputSchema).invoke([systemPrompt, new HumanMessage(humanMessageContent)])

	dispatchCustomEvent('reasoning', `${response.thought}\n\n`)

	if(response.nextAction === 'ask_user'){
		dispatchCustomEvent('final', response.responseToUser)
		return new Command({goto: END, update: {
				messages: [new AIMessage(response.responseToUser)],
				nextAction: response.nextAction
			}})
	} else if (
		response.nextAction === 'finish'
	){
		return new Command({goto: 'finalResponse', update: {
				nextAction: response.nextAction,
				planner_thought: response.thought,
			}})
	} else if(state.turn_count > MAX_TURNS){
		dispatchCustomEvent('final', '对不起，我无法在10轮内完成规划。为了避免资源浪费，我将结束对话。请尝试更具体的请求。')
		return new Command({goto: END, update: {
				messages: [new AIMessage("对不起，我无法在10轮内完成规划。请尝试更具体的请求。")],
			}})
	}
	else {
		return new Command({goto: ['toolExecutor'], update: {
				nextAction: response.nextAction,
				tool_calls: response.tool_calls,
				turn_count: state.turn_count + 1,
			}})
	}
}

// 模拟百度搜索API
async function baiduSearch(args) {
	console.log(`[工具调用] 百度搜索: ${args.query}`);
	if (args.query.includes('停车')) {
		return `关于“${args.query}”的搜索结果：目的地停车场有3个，但节假日期间可能紧张。建议提前出发。`;
	}
	return `关于“${args.query}”的搜索结果：以下是4个适合北京4岁娃周日遛娃的地点：
- **阿派朗创造力公园**：位于通州城市绿心森林公园，有8大板块，科技感十足，全龄适用，可休息、野餐。
- **冬奥公园马拉松大本营**：儿童游乐设施丰富且免费，有42公里步道，适合跑步骑行，对童车友好，停车也免费。
- **世界公园**：有小型动物园可喂动物，还有海底小纵队乐园，以及世界微型景观，能让孩子认知世界。
- **宋庆龄青少年科技文化交流中心**：一楼“启空间”有海洋球池、角色扮演区等，适合低龄宝宝；二楼“创空间”有机器人互动等，能让孩子边玩边学。`;
}

// 模拟和风天气API
async function qWeather(args) {
	console.log(`[工具调用] 和风天气: 查询 ${args.location} 在 ${args.date} 的天气`);
	return `天气预报：${args.location} 在 ${args.date} 全天晴朗，气温25-32摄氏度，无特殊天气。`;
}

// 模拟高德地图API
async function gaodeMaps(args) {
	console.log(`[工具调用] 高德地图: 从 ${args.start} 到 ${args.end}`);
	return `导航信息：从 ${args.start} 到 ${args.end} 预计需要1小时45分钟，当前路况良好。`;
}

const toolRegistry = {
	baidu_search: baiduSearch,
	qweather: qWeather,
	gaode_maps: gaodeMaps
}

const toolExecutorNode = async (state) =>{
	console.log('Enter tool executor');

	if (!state.tool_calls || state.tool_calls.length === 0) {
		return { tool_outputs: [] };
	}
	// 1. 为每个工具调用创建一个执行Promise
	const promises = state.tool_calls.map(async (toolCall) => {
		const { toolName, toolArgs } = toolCall;
		const toolFunction = toolRegistry[toolName];

		if (!toolFunction) {
			return { toolName, output: `错误：工具 "${toolName}" 不存在。` };
		}

		try {
			const output = await toolFunction(toolArgs);
			return { toolName, output };
		} catch (error) {
			return { toolName, output: `错误：执行工具 "${toolName}" 失败 - ${error.message}` };
		}
	});

	// 2. 使用 Promise.all 并发执行所有工具调用
	const results = await Promise.all(promises);

	console.log(`---[所有工具并发执行完毕，返回结果列表]---\n`, results);

	// 3. 返回包含所有结果的列表
	return { tool_outputs: results };
}


const summarizerNode = async(state) =>{
	console.log('Enter Summarizer Node');
	const summarizerPromptTemplate = PromptTemplate.fromTemplate(
		// ... 将上面第一部分的核心提示词粘贴到这里 ...
		`
# ROLE
你是一个高效的对话摘要引擎。
...
【现有摘要】:
{existing_summary}

【最新对话】:
{new_dialogue}

【你的输出】:
`
	);
	const {messages, summary} = state;
	const newDialogueText = messages.slice(-5).map((msg)=>`${msg.role}: ${msg.content}`).join('\n\n---\n\n');
	const newSummary = summarizerPromptTemplate
		.pipe(llm)
		.pipe(new StringOutputParser())
		.invoke({
			existing_summary: summary,
			new_dialogue: newDialogueText
		})
	console.log(newSummary)
	return {summary: newSummary};
}

// const baiduTest = async (state) => {
// 	console.log('Enter baiduTest');
// 	console.log(state.scenic);
// 	const baidu = new ChatOpenAI({
// 		modelName: 'ernie-3.5-8k',
// 		streaming: true,
// 		apiKey: state.modelConfig.baiduToken,
// 		configuration: {
// 			baseURL: "https://qianfan.baidubce.com/v2/ai_search/",
// 		},
// 	})
// 	const response = await baidu.withConfig({tags: ['final']}).invoke([new HumanMessage(`${state.scenic}怎么样？`)])
// 	return {
// 		messages: [response]
// 	}
// }

const finalResponseNode = async(state)=>{
	console.log('Enter Final Response Node');
	const finalResponsePromptTemplate = new PromptTemplate({
		template: `
# ROLE (角色定义)
你是一位名为“AI亲子旅行规划师”的AI助手。你亲切、热情、细致入博，并且非常擅长将复杂的信息整合成清晰、引人入胜的旅行计划。

# TASK (任务指令)
你的任务是根据下面提供的【背景信息清单】，为用户生成一份完整、生动、易于阅读的最终旅行计划。你的语气应该是鼓励和令人兴奋的，就像一位正在为好朋友介绍完美周末的朋友一样。请将所有相关信息都巧妙地编织到你的回复中，不要只是简单地罗列数据。

# FORMATTING (格式要求)
- **必须使用Markdown格式** 来组织你的回答，使其层次分明。
- **使用清晰的标题**，例如 \`## 您的专属旅行计划 🚗\`，\`### 目的地概览 🌳\` 等。
- **使用项目符号（bullet points）** 来列出具体的行程安排、注意事项或打包建议。
- **适当地使用表情符号 (emojis)** 来增加趣味性和可读性，这趟旅程是为了带孩子出去玩！
- **将关键信息加粗**，例如 **预计时长** 或 **天气提醒**。

---
# BACKGROUND INFORMATION CHECKLIST (背景信息清单)

### 1. 用户的核心需求 (User's Core Request)
{userInput}

### 2. 整体对话摘要 (Conversation Summary)
{summary}

### 3. 用户的历史偏好 (User's Historical Preferences)
{memory}

### 4. 所有工具的查询结果 (Collected Information from Tools)
{tool_outputs}

### 5. 你的最终思考 (Your Final Thought)
{planner_thought}
---

现在，请开始撰写你的最终旅行计划吧！
  `,
		inputVariables: [
			"userInput",
			"summary",
			"memory",
			"tool_outputs",
			"planner_thought",
		],});

	const formattedToolOutputs = state.tool_outputs
		.map(r => `[工具: ${r.toolName}]\n${r.output}`)
		.join('\n\n');

	const finalChain = finalResponsePromptTemplate
		.pipe(llm)
		.pipe(new StringOutputParser());

	const finalResponse = await finalChain.withConfig({tags: ['final']}).invoke({
		userInput: state.userInput,
		summary: state.summary,
		memory: state.memory,
		tool_outputs: formattedToolOutputs,
		planner_thought: state.planner_thought
	})

	return {
		messages: [finalResponse]
	}
}



export async function* main(input, modelConfig){
	const workflow = new StateGraph(ParentState)
		.addNode('agent', entry)
		.addNode('memoryRetriever', memoryRetrieverNode)
		.addNode('planner', plannerNode, {ends: ['toolExecutor', 'finalResponse', END]})
		.addNode('toolExecutor', toolExecutorNode)
		.addNode('finalResponse', finalResponseNode)
		.addNode('summarizer', summarizerNode)
		.addEdge(START, 'agent')
		.addEdge('agent', 'memoryRetriever')
		.addEdge('memoryRetriever', 'planner')
		.addEdge('toolExecutor', 'planner')
		.addEdge('finalResponse', 'summarizer')
		.addEdge('summarizer', END)

	const app = workflow.compile()

	const events = app.streamEvents({messages: input, modelConfig: modelConfig}, {subgraphs: true, version: 'v2'})

	for await (const event of events){

		if(event.event === 'on_chat_model_stream' && event.data?.chunk?.content){
			if(event.tags?.includes('final')){
				yield event.data.chunk
			} else if (event.tags?.includes('reasoning')) {
				yield {reasoning: event.data.chunk.content}
			}
			// console.log(event.data?.chunk?.content)
		}

		if (event.event === 'on_custom_event') {
			if(event.name === 'reasoning'){
				yield {reasoning: event.data}
			} else if(event.name === 'final'){
				yield {content: event.data}
			}
		}

		// if(event.event === 'on_chat_model_stream' && event.data?.chunk?.content){
		// 	console.log(event.data?.chunk?.content)
		// }
	}
}


