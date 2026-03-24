# /// script
# requires-python = ">=3.10"
# dependencies =[
#     "openai",
#     "transformers",
#     "tokenizers",
#     "jinja2"
# ]
# ///

import os
import argparse
import sys
import time
from urllib.parse import urlparse
from collections.abc import Mapping

import httpx
from openai import DefaultHttpxClient, OpenAI
from tests.chat_test_support import merge_reasoning_text, parse_content_text

DEFAULT_MAX_COMPLETION_TOKENS = 6000


def get_tokenizer(tokenizer_name_or_path: str):
    """加载 HuggingFace Tokenizer"""
    print(f"[*] 正在加载 Tokenizer: {tokenizer_name_or_path}")
    from transformers import AutoTokenizer
    # 启用 trust_remote_code 以支持像 Qwen 等自定义 Tokenizer 逻辑
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    return tokenizer

def normalize_vllm_base_url(raw_url: str) -> str:
    """Normalize vLLM OpenAI-compatible base_url.

    vLLM typically serves OpenAI-compatible routes under `/v1`.
    The OpenAI Python SDK expects `base_url` to include that prefix.
    """
    url = raw_url.strip()
    if not url:
        return url
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url

    parsed = urlparse(url)
    # If the caller passed only scheme://host[:port] (no path), append `/v1`.
    if parsed.path in ("", "/"):
        return f"{url}/v1"

    return url


def build_openai_client(*, base_url: str, api_key: str, no_proxy: bool) -> OpenAI:
    http_client = None
    if no_proxy:
        http_client = DefaultHttpxClient(trust_env=False)

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        timeout=httpx.Timeout(60.0),
        max_retries=0,
    )


def print_proxy_env() -> None:
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
    values = {key: os.getenv(key) for key in proxy_keys if os.getenv(key)}
    if not values:
        return
    print("[*] 检测到代理环境变量 (httpx/openai SDK 默认会读取):")
    for key, value in values.items():
        print(f"    - {key}={value}")


def probe_models_endpoint(client: OpenAI) -> None:
    try:
        result = client.models.list(timeout=15.0)
    except Exception as exc:
        print(f"[!] /v1/models 探测失败: {exc}")
        print("    提示: 如果你在访问内网地址(如 172.16.x.x)，请确保代理被绕过：")
        print("          1) 设置 NO_PROXY=172.16.84.27 或 NO_PROXY=172.16.0.0/16")
        print("          2) 或使用本脚本的 --no-proxy 参数")
        raise

    ids = []
    for item in getattr(result, "data", []) or []:
        model_id = getattr(item, "id", None)
        if isinstance(model_id, str):
            ids.append(model_id)
        if len(ids) >= 5:
            break

    if ids:
        print(f"[*] /v1/models 探测成功 (示例前 5 个): {', '.join(ids)}")
    else:
        print("[*] /v1/models 探测成功 (但未解析到 model id)")


def _count_chat_template_tokens(tokenizer, messages: list[dict[str, str]]) -> int:
    """Count prompt tokens using tokenizer chat template if supported."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except TypeError:
        # Older tokenizers may not support `tokenize=`; fall back to plain encoding.
        raise

    if isinstance(encoded, Mapping) and "input_ids" in encoded:
        input_ids = encoded["input_ids"]
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        if isinstance(input_ids, list):
            return len(input_ids)

    if isinstance(encoded, (list, tuple)):
        return len(encoded)

    if hasattr(encoded, "shape"):
        # torch / numpy tensor (1, seq_len) or (seq_len,)
        shape = getattr(encoded, "shape", None)
        if shape and len(shape) >= 1:
            return int(shape[-1])

    return len(encoded)  # best-effort fallback


def _count_total_tokens(tokenizer, messages: list[dict[str, str]], use_chat_template: bool) -> int:
    if use_chat_template:
        return _count_chat_template_tokens(tokenizer, messages)

    merged_text = "".join(message["content"] for message in messages)
    return len(tokenizer.encode(merged_text, add_special_tokens=True))


def _build_long_filler_text(
    tokenizer,
    *,
    system_prompt: str,
    instruction_prompt: str,
    target_tokens: int,
    use_chat_template: bool,
) -> tuple[str, int]:
    """Build filler text by calibrating against the tokenizer's real token count."""
    filler_unit = " data"

    def total_tokens(repeats: int) -> int:
        content = (filler_unit * repeats) + instruction_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return _count_total_tokens(tokenizer, messages, use_chat_template)

    low = 0
    high = 1
    while total_tokens(high) < target_tokens:
        low = high
        high *= 2

    while low + 1 < high:
        middle = (low + high) // 2
        if total_tokens(middle) < target_tokens:
            low = middle
        else:
            high = middle

    best_repeats = high if total_tokens(high) <= target_tokens else low
    filler_text = filler_unit * best_repeats
    actual_tokens = total_tokens(best_repeats)

    return filler_text, actual_tokens


def _extract_text_response(message) -> str | None:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        visible_content = parse_content_text(content).visible_text
        return visible_content or None

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
        merged = parse_content_text("".join(text_parts)).visible_text
        return merged or None

    return None


def _normalize_optional_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _extract_reasoning_response(message) -> str | None:
    reasoning = _normalize_optional_text(getattr(message, "reasoning", None))
    reasoning_content = _normalize_optional_text(getattr(message, "reasoning_content", None))
    content = getattr(message, "content", None)
    content_reasoning = parse_content_text(content).reasoning_text if isinstance(content, str) else None
    return merge_reasoning_text(reasoning, reasoning_content, content_reasoning)


def _extract_stream_delta_text(delta) -> str | None:
    return _extract_text_response(delta)


def _extract_stream_delta_reasoning(delta) -> str | None:
    reasoning = _normalize_optional_text(getattr(delta, "reasoning", None))
    reasoning_content = _normalize_optional_text(getattr(delta, "reasoning_content", None))
    content = getattr(delta, "content", None)
    content_reasoning = parse_content_text(content).reasoning_text if isinstance(content, str) else None
    return merge_reasoning_text(reasoning, reasoning_content, content_reasoning)


def _stream_long_context_response(
    client: OpenAI,
    model: str,
    messages_full: list[dict[str, str]],
) -> tuple[str | None, str | None, str | None, float | None, str | None]:
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    opened_reasoning = False
    opened_content = False
    first_token_latency = None
    first_token_channel = None

    print("[*] 已开启 stream，实时打印增量输出:")
    start_time = time.perf_counter()

    stream = client.chat.completions.create(
        model=model,
        messages=messages_full,
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
        temperature=0,
        timeout=600.0,
        stream=True,
    )

    for chunk in stream:
        for choice in getattr(chunk, "choices", []) or []:
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            reasoning_piece = _extract_stream_delta_reasoning(delta)
            if reasoning_piece:
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - start_time
                    first_token_channel = "reasoning"
                    print(f"[*] 首 token 延时: {first_token_latency:.2f}s (channel={first_token_channel})")
                reasoning_parts.append(reasoning_piece)
                if not opened_reasoning:
                    print("[stream reasoning] ", end="", flush=True)
                    opened_reasoning = True
                sys.stdout.write(reasoning_piece)
                sys.stdout.flush()

            content_piece = _extract_stream_delta_text(delta)
            if content_piece:
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - start_time
                    first_token_channel = "content"
                    print(f"[*] 首 token 延时: {first_token_latency:.2f}s (channel={first_token_channel})")
                answer_parts.append(content_piece)
                if opened_reasoning and not opened_content:
                    print()
                if not opened_content:
                    print("[stream content] ", end="", flush=True)
                    opened_content = True
                sys.stdout.write(content_piece)
                sys.stdout.flush()

    if opened_reasoning or opened_content:
        print()

    answer = "".join(answer_parts).strip() or None
    reasoning = "".join(reasoning_parts).strip() or None
    return answer, reasoning, finish_reason, first_token_latency, first_token_channel


def verify_long_context(client: OpenAI, model: str, tokenizer, target_tokens: int, stream: bool):
    print("\n" + "="*60)
    print(f" 验证 1: 测试超长上下文支持 (目标: {target_tokens} Tokens)")
    print("="*60)
    
    secret_code = "VLLM_CONTEXT_SUCCESS"
    system_prompt = "hi, You are a helpful assistant."
    instruction_prompt = f"\n\nBased on all the text above, the secret code is '{secret_code}'. What is the secret code? Just reply with the code."
    
    # 1. 首先计算基础 Prompt 和 Chat Template 占用的 Token 数
    messages_empty =[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction_prompt}
    ]
    
    try:
        base_len = _count_chat_template_tokens(tokenizer, messages_empty)
        use_chat_template = True
    except Exception as e:
        print(f"[!] 模型不支持 apply_chat_template，回退到基础 encode 计数: {e}")
        base_text = system_prompt + instruction_prompt
        base_len = len(tokenizer.encode(base_text, add_special_tokens=True))
        use_chat_template = False

    # 2. 计算需要填充的 Token 数量
    needed_tokens = target_tokens - base_len
    if needed_tokens <= 0:
        raise ValueError(f"目标 Token 数 ({target_tokens}) 太小，基础 prompt 已经占用了 {base_len} tokens。")
    
    print(f"[*] 基础模板占用: {base_len} tokens, 需填充无意义内容: {needed_tokens} tokens")
 
    # 3. 按真实 tokenizer 结果校准填充文本，避免 decode/re-encode 后发生 token 合并。
    print("[*] 正在校准生成填充字符串...")
    filler_text, calibrated_total_tokens = _build_long_filler_text(
        tokenizer,
        system_prompt=system_prompt,
        instruction_prompt=instruction_prompt,
        target_tokens=target_tokens,
        use_chat_template=use_chat_template,
    )
    
    # 4. 组装最终的 messages
    final_user_content = filler_text + instruction_prompt
    messages_full =[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_user_content}
    ]
    
    # 5. 再次精确验证总 Token 数
    actual_total_tokens = _count_total_tokens(tokenizer, messages_full, use_chat_template)
        
    print(f"[*] 实际构造的总请求 Tokens 数量: {actual_total_tokens}")
    if actual_total_tokens != calibrated_total_tokens:
        print(f"[!] 二次校验发现 token 数变化: 预估 {calibrated_total_tokens}, 实测 {actual_total_tokens}")
    if stream:
        print(f"[*] 正在发送流式请求... (处理 {actual_total_tokens} tokens 可能需要很久，请耐心等待)")
    else:
        print(f"[*] 正在发送请求... (处理 {actual_total_tokens} tokens 可能需要几分钟，请耐心等待)")
    
    try:
        if stream:
            answer, reasoning, finish_reason, first_token_latency, first_token_channel = _stream_long_context_response(
                client,
                model,
                messages_full,
            )
            print(f"[+] 流式请求完成！finish_reason={finish_reason}")
            if first_token_latency is not None:
                print(f"[*] 首 token 延时: {first_token_latency:.2f}s (channel={first_token_channel})")
            else:
                print("[!] 流式请求未观测到首个文本 token。")
            tool_calls = []
            refusal = None
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages_full,
                max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                temperature=0,
                timeout=600.0  # 10 分钟超时
            )
            choice = response.choices[0]
            message = choice.message
            answer = _extract_text_response(message)
            reasoning = _extract_reasoning_response(message)
            finish_reason = getattr(choice, "finish_reason", None)
            tool_calls = getattr(message, "tool_calls", None) or []
            refusal = getattr(message, "refusal", None)

            print(f"[+] 请求成功！finish_reason={finish_reason}")

        if not answer:
            print("[!] 响应中没有可提取的文本 content。")
            print(f"[*] tool_calls 数量: {len(tool_calls)}")
            if reasoning:
                print(f"[*] reasoning_content: {reasoning}")
            if refusal:
                print(f"[*] refusal: {refusal}")
            if reasoning:
                print("[-] 结论: 服务返回了 reasoning_content，但没有最终 answer content。")
            else:
                print("[-] 结论: 这次失败不是脚本构造了超长上下文，而是服务返回了空文本响应。")
            return

        print(f"[*] 模型回复: {answer}")
        if reasoning:
            print(f"[*] reasoning_content: {reasoning}")
        
        if secret_code in answer:
            print(f"[+] 结论: 完美支持 {target_tokens} 级别上下文，未发生 Lost in the middle 现象！")
        else:
            print("[-] 结论: 支持超长文本输入，但未能准确提取末尾的密码信息。")
            
    except Exception as e:
        print(f"[-] 请求失败: {e}")
        print("[-] 结论: 服务端报错，可能是 max_model_len 不足，或发生了显存 OOM 崩溃。")

def verify_tool_calling(client: OpenAI, model: str):
    print("\n" + "="*60)
    print(" 验证 2: 测试 Tool 调用 (Function Calling) 支持")
    print("="*60)
    
    tools =[
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. Boston, MA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    print("[*] 正在发送带有 get_current_weather tool 的天气查询请求...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What is the weather like in Boston today?"}
            ],
            tools=tools,
            tool_choice="auto",
            timeout=60.0
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            print("[+] 请求成功！模型成功触发了 Tool Calls:")
            for tool_call in message.tool_calls:
                print(f"    - 调用的函数名: {tool_call.function.name}")
                print(f"    - 传入的参数:   {tool_call.function.arguments}")
            print("[+] 结论: 服务支持 Tool 调用！")
        else:
            print(f"[-] 请求成功，但模型忽略了 Tool 而是返回了纯文本:")
            print(f"    {message.content}")
            print("[-] 结论: 接口正常，但模型自身不支持 Tool 调用或表现不佳。")
            
    except Exception as e:
        print(f"[-] 请求失败: {e}")
        print("[-] 结论: API 服务报错，当前 vLLM 引擎可能未开启/不支持该模型的 tool chat template。")

def main():
    parser = argparse.ArgumentParser(description="使用真实 Tokenizer 验证 vLLM OpenAI API 兼容性")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="vLLM API Base URL")
    parser.add_argument("--key", type=str, default="EMPTY", help="API Key (如果开启了鉴权)")
    parser.add_argument("--model", type=str, required=True, help="部署在 vLLM 上的模型名称 (用于 API 调用, 必填)")
    parser.add_argument("--tokenizer", type=str, default=None, help="HuggingFace Tokenizer 路径或名称 (默认同 --model)")
    parser.add_argument("--target-tokens", type=int, default=128000, help="测试的目标总 Token 数量 (默认: 128000)")
    parser.add_argument("--skip-long", action="store_true", help="跳过长上下文测试")
    parser.add_argument("--no-proxy", action="store_true", help="忽略 HTTP(S)_PROXY/ALL_PROXY 等环境变量 (适合内网地址)")
    parser.add_argument("--stream", action="store_true", help="长上下文验证使用 stream 输出并实时打印增量")
    
    args = parser.parse_args()

    base_url = normalize_vllm_base_url(args.url)
    if base_url != args.url.rstrip("/"):
        print(f"[*] base_url 规范化: {args.url} -> {base_url}")

    if not args.no_proxy:
        print_proxy_env()

    client = build_openai_client(base_url=base_url, api_key=args.key, no_proxy=args.no_proxy)
    
    print(f"=== 验证环境配置 ===")
    print(f"API URL: {base_url}")
    print(f"Model:   {args.model}")

    probe_models_endpoint(client)
    
    # 执行长上下文测试
    if not args.skip_long:
        tokenizer_path = args.tokenizer if args.tokenizer else args.model
        try:
            tokenizer = get_tokenizer(tokenizer_path)
            verify_long_context(client, args.model, tokenizer, args.target_tokens, args.stream)
        except ImportError:
            print("[!] 无法导入 transformers 库，请确保网络正常或手动安装。")
        except Exception as e:
            print(f"[!] Tokenizer 加载失败，跳过长上下文测试: {e}")
            print("    提示: 如果 '--model' 不是真实的 HuggingFace ID（例如是本地路径或别名），请使用 '--tokenizer' 参数显式指定真实的 Tokenizer 路径或 HF Repo ID。")
    else:
        print("\n[!] 已跳过长上下文测试。")
        
    # 执行 Tool Calling 测试
    verify_tool_calling(client, args.model)
    
    print("\n" + "="*60)
    print(" 验证结束")
    print("="*60)

if __name__ == "__main__":
    main()
