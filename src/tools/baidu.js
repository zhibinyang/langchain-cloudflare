import { ChatOpenAI } from '@langchain/openai';

export async function* search(queryString, config){
	// const res = await fetch(`mk4wctabd2.re.qweatherapi.com`, {
	// 	method: 'GET',
	// 	headers: {
	// 		'X-Appbuilder-Authorization': `Bearer ${config.BAIDU_API_KEY}`,
	// 		'Content-Type': 'application/json'
	// 	},
	// 	body: JSON.stringify({
	// 		messages: [
	// 			{
	// 				content: queryString,
	// 				role: 'user'
	// 			}
	// 		],
	// 		stream: true,
	// 		model: "ernie-3.5-8k",
	// 		enable_deep_search: false,
	// 		enable_followup_query: false,
	// 		resource_type_filter: [{"type": "web", "top_k":10}]
	// 	})
	// })
	//
	// for await (const chunk of res)
	//
	// return res.json()?.choices?.[0]?.message?.content


	const baidu = new ChatOpenAI({
		modelName: 'ernie-3.5-8k',
		streaming: true,
		apiKey: config.token,
		configuration: {
			baseURL: "https://qianfan.baidubce.com/v2/ai_search/",
		},
	})
}
