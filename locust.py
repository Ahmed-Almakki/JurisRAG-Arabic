import os
import random
import time

from locust import HttpUser, between, task

QUESTIONS = [
    "ما هي مدة تقادم الضرائب المستحقة للدولة؟",
    "ما الشروط القانونية لصحة عقد الإيجار؟",
    "ما هي حقوق العامل عند انتهاء عقد العمل؟",
    "What is the limitation period for claims owed to the state?",
    "What are the legal requirements for a valid lease agreement?",
    "What rights does an employee have when an employment contract ends?",
]


class JurisRagLLMPerformanceUser(HttpUser):
    """Simulate streaming requests and measure TTFT, ITL, and E2E Latency."""

    host = os.getenv("LOCUST_HOST", "http://localhost:3000")
    wait_time = between(10, 20)

    @task
    def ask_streaming_rag(self) -> None:
        question = random.choice(QUESTIONS)
        url = "/ask/"
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        payload = {"query": question}

        start_time = time.time()
        first_token_time = None
        prev_chunk_time = None
        token_count = 0
        itl_list = []

        try:
            with self.client.post(
                url, json=payload, headers=headers, stream=True, catch_response=True
            ) as response:
                if response.status_code >= 400:
                    response.failure(f"API returned HTTP {response.status_code}")
                    return

                # Iterate through streaming chunks (Server-Sent Events or chunked output)
                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        current_time = time.time()

                        if first_token_time is None:
                            first_token_time = current_time
                            # Calculate Time to First Token (TTFT) in milliseconds
                            ttft_ms = (first_token_time - start_time) * 1000

                            # Report TTFT as a custom metric to Locust UI
                            self.environment.events.request.fire(
                                request_type="POST",
                                name="POST /ask/ [TTFT]",
                                response_time=ttft_ms,
                                response_length=len(chunk),
                                exception=None,
                            )
                        else:
                            # Calculate Inter-Token Latency (ITL) between chunks
                            itl_ms = (current_time - prev_chunk_time) * 1000
                            itl_list.append(itl_ms)

                        prev_chunk_time = current_time
                        token_count += 1

                end_time = time.time()
                total_latency_ms = (end_time - start_time) * 1000

                if first_token_time is None:
                    response.failure("Streamed response was empty (no tokens received)")
                else:
                    response.success()

                    # Report End-to-End Latency to Locust UI
                    self.environment.events.request.fire(
                        request_type="POST",
                        name="POST /ask/ [E2E Latency]",
                        response_time=total_latency_ms,
                        response_length=0,
                        exception=None,
                    )

                    # Log detailed metrics to console for quick inspection
                    avg_itl = sum(itl_list) / len(itl_list) if itl_list else 0
                    print(
                        f"[Metrics] TTFT: {first_token_time - start_time:.3f}s | "
                        f"E2E: {total_latency_ms:.2f}ms | "
                        f"Tokens: {token_count} | "
                        f"Avg ITL: {avg_itl:.2f}ms"
                    )

        except Exception as e:  # noqa: BLE001
            # Report failure to Locust stats if connection drops or times out
            elapsed_ms = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="POST",
                name="POST /ask/ [Error]",
                response_time=elapsed_ms,
                response_length=0,
                exception=e,
            )
