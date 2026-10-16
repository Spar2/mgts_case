"""Write the gap shortlist. Two runs of this script write the same shortlist.json."""
import json

from fiber.shortlist import evidence_paths, write_shortlist


def main():
    text = write_shortlist()
    document = json.loads(text)
    print(f"path {evidence_paths()['shortlist']}")
    print(f"n_cards {document['n_cards']}")
    print(f"not_the_full_queue {document['not_the_full_queue']}")
    print(f"threshold_z {document['threshold_z']}")
    print("buckets " + json.dumps(document["buckets"], ensure_ascii=False, sort_keys=True))
    for card in document["cards"]:
        measured = card["measurements"]
        print(
            f"{card['id']} bucket={card['bucket']} record={card['record']} "
            f"width={measured.get('width_m')} contrast={measured.get('contrast_c')} "
            f"gap={measured.get('far_gap_c')} inbox={measured.get('in_box_count')} "
            f"alarm={measured.get('alarm_cells')} origin={measured.get('origin')}"
        )


if __name__ == "__main__":
    main()
