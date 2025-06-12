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
	HumanMessage
} from "@langchain/core/messages";
import {main as mainGraph} from './graphs';

import { v4 as uuidv4 } from 'uuid';

export default {
	async fetch(request, env, ctx) {
		const url = new URL(request.url)
		if (url.pathname !== "/v1/chat/completions") {
			return new Response("Not found", { status: 404 })
		}
		const requestId = uuidv4()
	  const requestObj = await request.json()
		console.log(requestObj)
		const { messages, stream = false, model = "doubao-seed-1-6-flash-250615", temperature = 0.7 } = requestObj
		const llm = new ChatOpenAI({
			modelName: model,
			temperature,
			streaming: stream,
			apiKey: env.OPENAI_API_KEY,
			configuration: {
				baseURL: "https://ark.cn-beijing.volces.com/api/v3/",
			},
		})

		const humanMessages = messages
			.filter((m) => m.role === "user")
			.map((m) => new HumanMessage(m.content))

		let response = '';
		if (!stream) {
			const res = await llm.invoke(humanMessages)
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
				const streamIterator = await mainGraph(humanMessages, {
					model: model,
					temperature: temperature,
					token: env.OPENAI_API_KEY,
					baiduToken: env.BAIDU_API_KEY
				})

				const created = Math.floor(Date.now() / 1000);

				for await (const chunk of streamIterator) {
					const token = chunk.content
					response += token
					const data = JSON.stringify({
						id: requestId,
						object: 'chat.completion.chunk',
						model: model,
						created,
						choices: [
							{
								delta: { content: token, role: 'assistant' },
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
