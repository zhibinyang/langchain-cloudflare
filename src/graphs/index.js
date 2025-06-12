import { Annotation, messagesStateReducer, StateGraph, START, END} from '@langchain/langgraph'
import { ChatOpenAI } from '@langchain/openai';

const ParentState = Annotation.Root({
	messages: Annotation({
		reducer: messagesStateReducer,
		default: () => []
	}),
	modelConfig: Annotation({})
});

const entry = async (state) =>{
	const llm = new ChatOpenAI({
		modelName: state.modelConfig.model,
		temperature: state.modelConfig.temperature,
		streaming: true,
		apiKey: state.modelConfig.token,
		configuration: {
			baseURL: "https://ark.cn-beijing.volces.com/api/v3/",
		},
	})
	const response = await llm.invoke(state.messages)
	return {
		messages: [response]
	}
}

export async function* main(input, modelConfig){
	const workflow = new StateGraph(ParentState)
		.addNode('agent', entry)
		.addEdge(START, 'agent')
		.addEdge('agent', END)

	const app = workflow.compile()

	const events = app.streamEvents({messages: input, modelConfig: modelConfig}, {subgraphs: true, version: 'v2'})

	for await (const event of events){
		// console.log(event)
		if(event.event === 'on_chat_model_stream' && event.data?.chunk?.content){
			yield event.data.chunk
		}
	}
}


