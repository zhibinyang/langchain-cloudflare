import { Annotation, messagesStateReducer, StateGraph, START, END, Command} from '@langchain/langgraph'
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod'
import { AIMessage } from '@langchain/core/messages';

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
	const schema = {
		intent: z.enum(['scenic_check', 'others']).default('others').describe('The intent' +
				' of' +
			' the user message'),
		response: z
			.string()
			.describe(
				'A human readable response to the original query, to acknowledge the intent.'
			),
		scenic: z.string().optional().describe('The name of the scenic spot, if applicable.'),
	};
	const response = await llm.withStructuredOutput(schema).invoke(state.messages)
	if(response.intent === 'others') {
		return new Command({goto: END, update: {
			messages: [new AIMessage(response.response)]
			}})
	} else {
		return new Command({goto: END, update: {
			messages: [new AIMessage('I will query the scenic spot: ' + response.scenic)],
			}})
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

		if(event.event === 'on_chat_model_stream' && event.data?.chunk?.content){
			yield event.data.chunk
		} else {
			if(event.event === 'on_chain_end' ){
				console.log(event)
				console.log(event.data?.output)
				console.log('===')
			}
		}
	}
}


