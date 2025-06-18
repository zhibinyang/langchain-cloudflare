import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';

export async function baiduSearch(args, state)  {
	console.log(`[工具调用] 百度搜索: ${args.query}`);
	const baidu = new ChatOpenAI({
		modelName: 'ernie-3.5-8k',
		streaming: true,
		apiKey: state.modelConfig.baiduToken,
		configuration: {
			baseURL: "https://qianfan.baidubce.com/v2/ai_search/",
		},
	})
	const response = await baidu.withConfig({tags: ['reasoning']}).invoke([new HumanMessage(args.query)])
	console.log(`百度搜索结果: ${response.content}`);
	return response.content
}
