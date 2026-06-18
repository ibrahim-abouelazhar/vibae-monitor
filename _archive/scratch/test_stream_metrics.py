import httpx
import asyncio
import json

async def main():
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Request stream for Time_Normal_1_098.mat
        async with client.stream("GET", "http://127.0.0.1:8000/predict/stream?file=Time_Normal_1_098.mat") as response:
            print("Status:", response.status_code)
            count = 0
            async for line in response.aiter_lines():
                if line:
                    print(line)
                    count += 1
                    if count > 200:
                        break

asyncio.run(main())
