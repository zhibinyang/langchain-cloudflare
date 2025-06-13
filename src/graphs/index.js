import { Annotation, messagesStateReducer, StateGraph, START, END, Command} from '@langchain/langgraph'
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod'
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import { search } from '../tools/baidu'

const ParentState = Annotation.Root({
	messages: Annotation({
		reducer: messagesStateReducer,
		default: () => []
	}),
	modelConfig: Annotation({}),
	scenic: Annotation({})
});

const entry = async (state) =>{
	console.log('Enter Entry Node');
	const llm = new ChatOpenAI({
		modelName: state.modelConfig.model,
		temperature: state.modelConfig.temperature,
		streaming: true,
		apiKey: state.modelConfig.token,
		configuration: {
			baseURL: "https://ark.cn-beijing.volces.com/api/v3/",
		},
	})
	const schema = z.object({
		intent: z.enum(['scenic_check', 'others']).describe('The intent  of the' +
			' user message， others is the default intent if no scenic spot is mentioned.'),
		response: z
			.string()
			.describe(
				'A human readable response to the original query, to acknowledge the intent.'
			),
		scenic: z.string().nullable().describe('The name of the scenic spot, if applicable.'),
	});
	const response = await llm.withStructuredOutput(schema).invoke(state.messages)
	console.log(response)
	if(response.intent === 'others') {
		await dispatchCustomEvent(
			'final',
			response.response
		)
		return new Command({goto: END, update: {
			messages: [new AIMessage(response.response)]
			}})
	} else {
		return new Command({goto: 'baidu', update: {
				messages: [new AIMessage('I will query the scenic spot: ' + response.scenic)],
				scenic: response.scenic
			}})
	}
}

const baiduTest = async (state) => {
	console.log('Enter baiduTest');
	console.log(state.scenic);
	const baidu = new ChatOpenAI({
		modelName: 'ernie-3.5-8k',
		streaming: true,
		apiKey: state.modelConfig.baiduToken,
		configuration: {
			baseURL: "https://qianfan.baidubce.com/v2/ai_search/",
		},
	})
	const response = await baidu.withConfig({tags: ['final']}).invoke([new HumanMessage(`${state.scenic}怎么样？`)])
	return {
		messages: [response]
	}
}

export async function* main(input, modelConfig){
	const workflow = new StateGraph(ParentState)
		.addNode('agent', entry, {ends: ['baidu', END]})
		.addNode('baidu', baiduTest)
		.addEdge(START, 'agent')
		.addEdge('baidu', END)

	const app = workflow.compile()

	const events = app.streamEvents({messages: input, modelConfig: modelConfig}, {subgraphs: true, version: 'v2'})

	for await (const event of events){

		if(event.event === 'on_chat_model_stream' && event.data?.chunk?.content && event.tags?.includes('final')){
			console.log(event.data?.chunk?.content)
			yield event.data.chunk
		}

		if (event.event === 'on_custom_event') {
			yield {content: event.data}
		}

		// if(event.event === 'on_chat_model_stream' && event.data?.chunk?.content){
		// 	console.log(event.data?.chunk?.content)
		// }
	}
}


