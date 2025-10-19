#!/usr/bin/env python3
"""Debug script to check SERP API response structure."""

import json
from src.mcp_seo.dataforseo.client import DataForSEOClient

def main():
    client = DataForSEOClient()

    print("Testing SERP API response structure...")
    print("=" * 80)

    # Get SERP results
    result = client.get_serp_results(
        keyword="python seo",
        location_code=2840,  # United States
        language_code="en",
        depth=10
    )

    print("\nFull API Response:")
    print(json.dumps(result, indent=2, default=str))

    # Inspect the structure
    if result.get("tasks"):
        task = result["tasks"][0]
        print("\n" + "=" * 80)
        print("Task structure:")
        print(f"  - task.keys(): {task.keys()}")

        if task.get("result"):
            print(f"  - task['result'] type: {type(task['result'])}")
            print(f"  - task['result'] length: {len(task['result'])}")

            if len(task['result']) > 0:
                first_result = task['result'][0]
                print(f"  - task['result'][0] type: {type(first_result)}")

                if isinstance(first_result, dict):
                    print(f"  - task['result'][0].keys(): {first_result.keys()}")
                else:
                    print(f"  - task['result'][0] value: {first_result}")

if __name__ == "__main__":
    main()
