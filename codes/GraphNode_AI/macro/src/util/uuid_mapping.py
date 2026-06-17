import json
import os


def map_uuid_to_graph(conversations_path, graph_path, output_path):
    """English documentation."""

    # English comment.
    if not os.path.exists(conversations_path):
        print(f"Error: text text text text - {conversations_path}")
        return
    if not os.path.exists(graph_path):
        print(f"Error: text text text text - {graph_path}")
        return

    # English comment.
    print(
        f"Loading files...\n - Conversations: {conversations_path}\n - Graph: {graph_path}"
    )

    try:
        with open(conversations_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)

        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON files: {e}")
        return

    # English comment.
    mapped_count = 0

    if "nodes" in graph_data:
        for node in graph_data["nodes"]:
            orig_id = node.get("orig_id")  # English comment.

            if orig_id and isinstance(orig_id, str) and orig_id.startswith("conv_"):
                try:
                    # English comment.
                    idx = int(orig_id.split("_")[1])

                    # English comment.
                    if 0 <= idx < len(conversations):
                        # English comment.
                        real_uuid = conversations[idx].get("id")

                        if real_uuid:
                            # English comment.
                            node["uuid"] = real_uuid
                            mapped_count += 1

                except (ValueError, IndexError):
                    # English comment.
                    continue
    else:
        print("Error: text text 'nodes' text text.")
        return

    # English comment.
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        print(f"\nSuccess! text {mapped_count}text text UUIDtext text.")
        print(f"text text text: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    # English comment.
    # English comment.
    CONVERSATIONS_FILE = "../input_data/Test_macro/conversations.json"
    TARGET_GRAPH_FILE = "output/postprocess/frontend_graph_postprocessed.json"

    # English comment.
    OUTPUT_FILE = "output/final/frontend_graph_macro_with_uuid.json"

    # English comment.
    map_uuid_to_graph(CONVERSATIONS_FILE, TARGET_GRAPH_FILE, OUTPUT_FILE)
