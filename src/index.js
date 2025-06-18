/**
 * Welcome to Cloudflare Workers! This is your first worker.
 *
 * - Run `npm run dev` in your terminal to start a development server
 * - Open a browser tab at http://localhost:8787/ to see your worker in action
 * - Run `npm run deploy` to publish your worker
 *
 * Learn more at https://developers.cloudflare.com/workers/
 */

import { ChatOpenAI } from "@langchain/openai";
import {
	HumanMessage,
	AIMessage,
	SystemMessage,
	BaseMessage
} from "@langchain/core/messages";
import {main as mainGraph} from './graphs';

import { v4 as uuidv4 } from 'uuid';

export default {
	async fetch(request, env, ctx) {
		const url = new URL(request.url)
		const inboundAuthorization = request.headers.get("Authorization");
		let inboundToken = null
		if(inboundAuthorization) {
			const parts = inboundAuthorization.split(' ')
			if (parts.length === 2 && parts[0].toLowerCase() === 'bearer') {
				inboundToken = parts[1]
			}
		}

		if (url.pathname !== "/v1/chat/completions") {
			return new Response("Not found", { status: 404 })
		}
		const requestId = uuidv4()
	  const requestObj = await request.json()
		console.log(requestObj)
		const { messages, stream = false, model = "doubao-seed-1-6-250615", temperature = 0.7 } = requestObj
		const llm = new ChatOpenAI({
			modelName: model,
			temperature,
			streaming: stream,
			apiKey: env.OPENAI_API_KEY,
			configuration: {
				baseURL: "https://ark.cn-beijing.volces.com/api/v3/",
			},
		})

		// const humanMessages = messages
		// 	.filter((m) => m.role === "user")
		// 	.map((m) => new HumanMessage(m.content))

		const langchainMessages = messages.map((m) => {
			switch (m.role) {
				case "user":
					return new HumanMessage(m.content);
				case "assistant":
					return new AIMessage(m.content);
				case "system":
					return new SystemMessage(m.content);
				default:
					// 你可以为其他 role 执行不同逻辑：
					return new BaseMessage({ role: m.role, content: m.content });
			}
		});

		let response = '';
		if (!stream) {
			const res = await llm.invoke(langchainMessages)
			return Response.json({
				id: requestId,
				object: "chat.completion",
				created: Math.floor(Date.now() / 1000),
				model,
				choices: [{
					index: 0,
					message: { role: "assistant", content: res },
					finish_reason: "stop",
				}],
			})
		}

		// ✅ SSE Streaming
		const streamSSE = new ReadableStream({
			async start(controller) {
				const encoder = new TextEncoder()

				// const streamIterator = await llm.stream(humanMessages)
				const streamIterator = await mainGraph(langchainMessages, {
					model: model,
					temperature: temperature,
					token: env.OPENAI_API_KEY,
					baiduToken: env.BAIDU_API_KEY,
					gaodeToken: env.GAODE_API_KEY,
					qweatherToken: env.QWEATHER_API_KEY,
					inboundToken: inboundToken,
					kvStore: env.USER_KV,
					vectorStore: env.VECTORIZE,
					ai: env.AI
				})

				const created = Math.floor(Date.now() / 1000);

				for await (const chunk of streamIterator) {
					const token = chunk.content
					const reasoning = chunk.reasoning
					response += token
					const data = JSON.stringify({
						id: requestId,
						object: 'chat.completion.chunk',
						model: model,
						created,
						choices: [
							{
								delta: { content: token || '',  reasoning_content: reasoning || '', role: 'assistant' },
								index: 0
							}
						]
					})
					controller.enqueue(encoder.encode(`data: ${data}\n\n`))
					await new Promise(r => setTimeout(r, 0));
				}
				const doneChunk = JSON.stringify({
					id: requestId,
					object: 'chat.completion.chunk',
					mode: model,
					choices: [
						{
							delta: {content:'',role:'assistant'},
							index: 0,
							finish_reason: 'stop'
						}
					]
				})
				controller.enqueue(encoder.encode(`data: ${doneChunk}\n\n`))
				controller.enqueue(encoder.encode(`data: [DONE]\n\n`))
				controller.close()
			},
		})

		return new Response(streamSSE, {
			headers: {
				"Content-Type": "text/event-stream",
				"Cache-Control": "no-cache",
				"Connection": "keep-alive",
				"Transfer-Encoding": "chunked",
				"X-Request-Id": requestId,
			},
		})
	}
};
