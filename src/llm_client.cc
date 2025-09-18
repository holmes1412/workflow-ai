#include <stdint.h>
#include "workflow/HttpMessage.h"
#include "workflow/HttpUtil.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/HttpMessage.h"
#include "workflow/WFHttpChunkedClient.h"
#include "workflow/Workflow.h"
#include "workflow/WFFacilities.h"
#include "workflow/WFFuture.h"
#include "llm_client.h"
#include "llm_session.h"

using namespace wfai;

static constexpr const char *default_url = "https://api.deepseek.com/v1/chat/completions";
static constexpr const char *auth_str = "Bearer ";
static constexpr uint32_t default_streaming_ttft = 100 * 1000; // ms
static constexpr uint32_t default_streaming_tpft = 1 * 1000; // ms
static constexpr uint32_t default_no_streaming_ttft = 500 * 1000; // ms
static constexpr uint32_t default_no_streaming_tpft = 100 * 1000; // ms
static constexpr int default_redirect_max = 3;

// for tool calls execution, both single or parallel
class ToolCallsData
{
public:

	~ToolCallsData()
	{
		for (auto *result : results)
			delete result;
	}

public:
	SessionContext *context;
	std::vector<FunctionResult *> results;
	std::vector<std::string> tool_call_ids;
};

// for streaming
// to collect the argument in tool_calls in each chunk
bool append_tool_call_from_chunk(const ChatCompletionChunk& chunk,
								 ChatCompletionResponse *resp)
{
	if (resp->choices.empty()) // first time to mark
	{
		ChatResponse::Choice choice;
		choice.message.tool_calls.push_back(chunk.choices[0].delta.tool_calls[0]);
		resp->choices.push_back(choice);
		return true;
	}

	// not the first time
	if (resp->choices[0].message.tool_calls.empty())
		return false;

	resp->choices[0].message.tool_calls[0].function.arguments +=
		chunk.choices[0].delta.tool_calls[0].function.arguments;

	return true;
}

LLMClient::LLMClient() :
	LLMClient("", default_url)
{
}

LLMClient::LLMClient(const std::string& api_key) :
	LLMClient(api_key, default_url)
{
}

LLMClient::LLMClient(const std::string& api_key,
					 const std::string& base_url) :
	api_key(api_key),
	base_url(base_url)
{
	this->redirect_max = default_redirect_max;
	this->streaming_ttft = default_streaming_ttft;
	this->streaming_tpft = default_streaming_tpft;
	this->ttft = default_no_streaming_ttft;
	this->tpft = default_no_streaming_tpft;
	this->function_manager = nullptr;
}

WFHttpChunkedTask *LLMClient::create_chat_task(ChatCompletionRequest& request,
											   llm_extract_t extract,
											   llm_callback_t callback)
{
	ChatCompletionRequest *req = new ChatCompletionRequest(request);
	ChatCompletionResponse *resp = new ChatCompletionResponse();

	SessionContext *ctx = new SessionContext(
		req, resp, std::move(extract), std::move(callback), true);

	return this->create(ctx);
}

WFHttpChunkedTask *LLMClient::create(SessionContext *ctx)
{
	auto extract_handler = std::bind(
		&LLMClient::extract,
		this,
		std::placeholders::_1,
		ctx
	);

	callback_t callback_handler;

	if (this->function_manager && ctx->req->tool_choice != "none")
	{
		auto tools = this->function_manager->get_functions();
		for (const auto& tool : tools)
			ctx->req->tools.emplace_back(tool);

		callback_handler = std::bind(
			&LLMClient::callback_with_tools,
			this,
			std::placeholders::_1,
			ctx
		);
	}
	else
	{
		callback_handler = std::bind(
			&LLMClient::callback,
			this,
			std::placeholders::_1,
			ctx
		);
	}

	auto *task = client.create_chunked_task(
		this->base_url,
		this->redirect_max,
		std::move(extract_handler),
		std::move(callback_handler)
	);

	if (ctx->req->stream)
	{
		task->set_watch_timeout(this->streaming_ttft);
		task->set_receive_timeout(this->streaming_tpft);
	}
	else
	{
		task->set_watch_timeout(this->ttft);
		task->set_receive_timeout(this->tpft);
	}

	auto *http_req = task->get_req();
	http_req->add_header_pair("Authorization", auth_str + this->api_key);
	http_req->add_header_pair("Content-Type", "application/json");
	http_req->add_header_pair("Connection", "keep-alive");
	http_req->set_method("POST");

	std::string body = ctx->req->to_json();
	http_req->append_output_body(body.data(), body.size());

	return task;
}

void LLMClient::callback(WFHttpChunkedTask *task, SessionContext *ctx)
{
	const void *body;
	size_t len;

	if (task->get_state() == WFT_STATE_SUCCESS && !ctx->req->stream)
	{
		if (ctx->resp->buffer_empty())
		{
			task->get_resp()->get_parsed_body(&body, &len);
			ctx->resp->ChatResponse::parse_json((const char *)body, len);
		}
		else
			ctx->resp->parse_json();
	}
	// TODO: if (!ret) set error

	if (ctx->callback)
		ctx->callback(task, ctx->req, ctx->resp);

	delete ctx;
}

void LLMClient::callback_with_tools(WFHttpChunkedTask *task,
									SessionContext *ctx)
{
	ChatCompletionRequest *req = ctx->req;
	ChatCompletionResponse *resp = ctx->resp;
	const void *body;
	size_t len;
	bool ret = true; // TODO: let's take streaming parse_json return true

	// for streaming:
	// 	already parse chunk and fill resp in append_tool_call_from_chunk()
	// for non streaming:
	// 	need to parse resp here
	if (task->get_state() == WFT_STATE_SUCCESS && !ctx->req->stream)
	{
		if (resp->buffer_empty())
		{
			task->get_resp()->get_parsed_body(&body, &len);
			ret = resp->ChatResponse::parse_json((const char *)body, len);
		}
		else
		{
			ret = resp->parse_json();
		}
	}

	// parse resp
	if (task->get_state() != WFT_STATE_SUCCESS ||
		!ret ||
		resp->choices.empty() ||
		resp->choices[0].message.tool_calls.empty())
	{
		if (ctx->callback)
			ctx->callback(task, req, resp);

		delete ctx;
		return;
	}

	// append the llm response
	Message resp_msg;
	resp_msg.role = "assistant";

	for (const auto &tc : resp->choices[0].message.tool_calls)
		resp_msg.tool_calls.push_back(tc);

	req->messages.push_back(resp_msg);

	// remove previous tools infomation
	req->tool_choice = "none";
	req->tools.clear();

	ToolCallsData *tc_data = new ToolCallsData();
	bool mgr_ret = false;

	// calculate
	if (resp->choices[0].message.tool_calls.size() == 1)
	{
		const auto& tc = resp->choices[0].message.tool_calls[0];
		// if (tc.type == "function")
		FunctionResult *res = new FunctionResult();
		tc_data->results.push_back(res);
		tc_data->tool_call_ids.push_back(tc.id);

		WFGoTask *next = this->function_manager->async_execute(
			tc.function.name,
			tc.function.arguments,
			res);

		if (next) // should return WFEmptyTask instead of nullptr
		{
			mgr_ret = true;
			next->user_data = tc_data;

			auto callback_handler = std::bind(
				&LLMClient::tool_calls_callback,
				this,
				std::placeholders::_1,
				ctx
			);

			next->set_callback(std::move(callback_handler));
			series_of(task)->push_front(next);
		}
	}
	else
	{
		auto p_cb = std::bind(
			&LLMClient::p_tool_calls_callback,
			this,
			std::placeholders::_1,
			ctx
		);

		ParallelWork *pwork = Workflow::create_parallel_work(std::move(p_cb));
		pwork->set_context(tc_data);

		// Create series for each tool call
		for (const auto& tc : resp->choices[0].message.tool_calls)
		{
			FunctionResult *res = new FunctionResult();
			tc_data->results.push_back(res);
			tc_data->tool_call_ids.push_back(tc.id);

			WFGoTask *go_task = this->function_manager->async_execute(
				tc.function.name,
				tc.function.arguments,
				res);

			if (go_task)
			{
				SeriesWork *series = Workflow::create_series_work(go_task, nullptr);
				pwork->add_series(series);
			}
		}

		series_of(task)->push_front(pwork);
	}

	if (!mgr_ret)
	{
		resp->state = RESPONSE_TOOLS_ERROR;

		if (ctx->callback)
			ctx->callback(task, req, resp);

		delete tc_data;
		delete ctx;
	}
}

void LLMClient::p_tool_calls_callback(const ParallelWork *pwork,
									  SessionContext *ctx)
{
	ToolCallsData *tc_data = static_cast<ToolCallsData *>(pwork->get_context());

	// Add all tool call results to the request messages
	for (size_t i = 0; i < tc_data->results.size(); ++i)
	{
		Message msg;
		msg.role = "tool";
		msg.tool_call_id = tc_data->tool_call_ids[i];
		if (tc_data->results[i]->success)
			msg.content = tc_data->results[i]->result;
		else
			msg.content = tc_data->results[i]->error_message;
		ctx->req->messages.push_back(std::move(msg));
	}

	ctx->resp->clear(); // clear resp for next round

	auto *next = this->create(ctx);
	series_of(pwork)->push_front(next);
	delete tc_data;
}

void LLMClient::tool_calls_callback(WFGoTask *task, SessionContext *ctx)
{
	ToolCallsData *tc_data = static_cast<ToolCallsData *>(task->user_data);
//	if (!ctx || ctx->results.empty() || ctx->tool_call_ids.empty())
//		return;

	Message msg;
	msg.role = "tool";
	msg.tool_call_id = tc_data->tool_call_ids[0];
	if (tc_data->results[0]->success)
		msg.content = tc_data->results[0]->result;
	else
		msg.content = tc_data->results[0]->error_message;
	ctx->req->messages.push_back(std::move(msg));

	ctx->resp->clear(); // clear resp for next round

	auto *next = this->create(ctx);
	series_of(task)->push_front(next);
	delete tc_data;
}

void LLMClient::extract(WFHttpChunkedTask *task, SessionContext *ctx)
{
	protocol::HttpMessageChunk *msg_chunk = task->get_chunk();
	const void *msg;
	size_t size;

	if (!msg_chunk->get_chunk_data(&msg, &size))
	{
		// TODO : mark error : invalid chunk data
		return;
	}

	if (!ctx->req->stream)
	{
		ctx->resp->append_buffer(static_cast<const char*>(msg), size);

		if (ctx->extract)
			ctx->extract(task, ctx->req, nullptr); // not a chunk for no streaming
	}
	else
	{
		const char *p = static_cast<const char *>(msg);
		const char *msg_end = p + size;
		const char *begin;
		const char *end;
		size_t len;

		while (p < msg_end)
		{
			begin = strstr(p, "data: ");
			if (!begin || begin >= msg_end)
				break;

			begin += 6;

			end = strstr(begin, "data: "); // \r\n
			end = end ? end : msg_end;
			p = end;

			while (end > begin && (*(end - 1) == '\n' || *(end - 1) == '\r'))
				--end;

			len = end - begin;
			if (len > 0)
			{
				ChatCompletionChunk chunk;
				if (chunk.parse_json(begin, len))
				{
					if (!chunk.choices.empty() &&
						!chunk.choices[0].delta.tool_calls.empty())
					{
						if (!append_tool_call_from_chunk(chunk, ctx->resp))
						{
							chunk.state = RESPONSE_FRAMEWORK_ERROR;
						}
					}

					if (ctx->extract)
					{
						ctx->extract(task, ctx->req, &chunk);
					}
					else if (ctx->is_async_streaming())
					{
						// created by async api and streaming mode
						if (!chunk.choices.empty() &&
							!chunk.choices[0].finish_reason.empty())
						{
							chunk.set_last_chunk(true);
						}

						ChatCompletionChunk *msg =
							new ChatCompletionChunk(std::move(chunk));

						ctx->async_msgqueue_put(msg);

						if (msg->last_chunk() == true)
							ctx->async_msgqueue_set_nonblock();
					}
				}
			}
		}
	}
}

void LLMClient::set_function_manager(FunctionManager *manager)
{
	this->function_manager = manager;
}

bool LLMClient::register_function(const FunctionDefinition& def,
								  FunctionHandler handler)
{
	return this->function_manager->register_function(def, std::move(handler));
}

void LLMClient::sync_callback(WFHttpChunkedTask *task,
							  ChatCompletionRequest *req, // useless
							  ChatCompletionResponse *resp, // from llm_callback_t
							  WFPromise<SyncResult> *promise)
{
	SyncResult result;

	if (task->get_state() != WFT_STATE_SUCCESS)
	{
		result.success = false;
		result.error_message = "Task execution failed. State: " +
			std::to_string(task->get_state()) +
			", Error: " + std::to_string(task->get_error());
	}
	else
	{
		protocol::HttpResponse *http_resp = task->get_resp();
		result.status_code = atoi(http_resp->get_status_code());
		if (result.status_code != 200)
		{
			result.success = false;
			result.error_message = "HTTP error: " +
				std::string(http_resp->get_status_code()) +
				" " + http_resp->get_reason_phrase();
		}
		else
		{
			result.success = true;
			result.response = std::move(*resp); // must move if need resp after callback
		}
	}

	promise->set_value(std::move(result));
	delete promise;
}

SyncResult LLMClient::chat_completion_sync(ChatCompletionRequest& request,
										   ChatCompletionResponse& response)
{
	auto *promise = new WFPromise<SyncResult>();
	auto future = promise->get_future();

	auto cb_for_sync = std::bind(
		&LLMClient::sync_callback,
		this,
		std::placeholders::_1, // the task from llm_callback_t
		std::placeholders::_2,
		std::placeholders::_3,
		promise
	);

	SessionContext *ctx = new SessionContext(&request, &response,
											 nullptr, std::move(cb_for_sync),
											 false);

	auto *task = this->create(ctx);
	task->start();
	SyncResult result = future.get();

	if (result.success)
		response = std::move(result.response);

	return result;
}

void LLMClient::async_callback(WFHttpChunkedTask *task,
							   ChatCompletionRequest *req,
							   ChatCompletionResponse *resp,
							   SessionContext *ctx)
{
	AsyncResultPtr *result = ctx->get_async_result();

	if (task->get_state() != WFT_STATE_SUCCESS)
	{
		resp->state = RESPONSE_FRAMEWORK_ERROR;
		result->set_success(false);
		result->set_error_message("Task execution failed. State: " +
			std::to_string(task->get_state()) +
			", Error: " + std::to_string(task->get_error()));
	}
	else
	{
		protocol::HttpResponse *http_resp = task->get_resp();

		int code = atoi(http_resp->get_status_code());
		result->set_status_code(code);

		if (code != 200)
		{
			resp->state = RESPONSE_NETWORK_ERROR;
			result->set_success(false);
			result->set_error_message("HTTP error: " +
				std::string(http_resp->get_status_code()) +
				" " + http_resp->get_reason_phrase());

		}
		else
		{
			resp->state = RESPONSE_SUCCESS;
			result->set_success(true);
		}
	}

	// user may waiting at get_chunk(), so use a chunk to send error
	if (resp->state != RESPONSE_SUCCESS && ctx->is_async_streaming())
	{
		auto chunk = new ChatCompletionChunk();
		chunk->state = resp->state;
		result->msg_queue_put(chunk);
	}

	// should this logic move inside AsyncResultPtr ?
	result->get_promise()->set_value(std::move(resp));
	delete result->get_promise();
	result->decref();
}

AsyncResult LLMClient::chat_completion_async(ChatCompletionRequest& request)
{
	ChatCompletionResponse *response = new ChatCompletionResponse();
	AsyncResult result;

	if (request.stream)
	{
		result.msg_queue_create(request.max_tokens + 2);
	}

	SessionContext *ctx = new SessionContext(&request, response,
											 nullptr, nullptr,
											 false);
	ctx->set_async_result(&result);

	auto cb_for_async = std::bind(
		&LLMClient::async_callback,
		this,
		std::placeholders::_1,
		std::placeholders::_2,
		std::placeholders::_3,
		ctx // TODO: this is not straight forward, but easy to manage
	);

	ctx->set_callback(std::move(cb_for_async));

	auto *task = this->create(ctx);
	task->start();

	return result;
}

