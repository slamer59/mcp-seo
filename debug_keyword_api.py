#!/usr/bin/env python3
"""
Debug script to check DataForSEO keyword API response
"""

import json

from src.mcp_seo.config.settings import get_language_code, get_location_code
from src.mcp_seo.dataforseo.client import DataForSEOClient


def main():
    # Initialize client
    client = DataForSEOClient()

    # Test keywords
    keywords = ["python code analyzer", "python dependency analysis"]
    location = "United States"
    language = "en"

    location_code = get_location_code(location)
    language_code = get_language_code(language)

    print(f"Testing keyword API with:")
    print(f"  Keywords: {keywords}")
    print(f"  Location: {location} (code: {location_code})")
    print(f"  Language: {language} (code: {language_code})")
    print("\n" + "=" * 80 + "\n")

    # Make API request
    print("Step 1: Creating keyword task...")
    result = client.get_keyword_data(
        keywords=keywords, location_code=location_code, language_code=language_code
    )

    print("Initial Response:")
    print(json.dumps(result, indent=2))
    print("\n" + "=" * 80 + "\n")

    if not result.get("tasks"):
        print("ERROR: No tasks in response!")
        return

    task_id = result["tasks"][0]["id"]
    print(f"Task ID: {task_id}")
    print(f"Task Status: {result['tasks'][0].get('status_message')}")
    print("\n" + "=" * 80 + "\n")

    # Wait for completion
    print("Step 2: Waiting for task completion...")
    completed_result = client.wait_for_task_completion(task_id, "keywords")

    print("Completed Response:")
    print(json.dumps(completed_result, indent=2))
    print("\n" + "=" * 80 + "\n")

    # Check result structure
    if completed_result.get("tasks"):
        task = completed_result["tasks"][0]
        print("Task Info:")
        print(f"  Status Code: {task.get('status_code')}")
        print(f"  Status Message: {task.get('status_message')}")
        print(f"  Cost: {task.get('cost')}")

        if task.get("result"):
            print(f"\nResult Info:")
            print(f"  Number of result entries: {len(task['result'])}")

            if task["result"]:
                first_result = task["result"][0]
                print(f"  First result keys: {list(first_result.keys())}")
                print(f"  Items count: {len(first_result.get('items', []))}")

                if first_result.get("items"):
                    print(f"\nFirst item sample:")
                    print(json.dumps(first_result["items"][0], indent=2))
                else:
                    print(f"\nWARNING: Items array is EMPTY!")
                    print(f"Full first result:")
                    print(json.dumps(first_result, indent=2))
        else:
            print("\nERROR: No result in task!")
    else:
        print("ERROR: No tasks in completed response!")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
