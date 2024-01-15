"""Analyze GitHub Actions run and provide feedback on pull request."""

import json
import os
import sys

from github import Github

# Retrieve necessary information from GitHub context
context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))
repo = context_dict["repository"]
g = Github(sys.argv[1])
repo = g.get_repo(repo)
pr_number = context_dict["event"]["number"]
pr = repo.get_pull(number=pr_number)

# Parse command line arguments
title_labels = sys.argv[2][1:-1].split(",")
title_labels_new = sys.argv[3][1:-1].split(",")
content_labels = sys.argv[4][1:-1].split(",")
content_labels_status = sys.argv[5]

# Perform replacements based on existing labels
replacement_labels = []
for i, label in enumerate(content_labels):
    for cur_label, new_label in replacement_labels:
        if label == cur_label:
            content_labels[i] = new_label

# Format labels for display in comment
labels = [(label.name, label.color) for label in repo.get_labels()]
title_labels = [
    "$\\color{#%s}{\\textsf{%s}}$" % (color, label)
    for label, color in labels
    if label in title_labels
]
title_labels_new = [
    "$\\color{#%s}{\\textsf{%s}}$" % (color, label)
    for label, color in labels
    if label in title_labels_new
]
content_labels = [
    "$\\color{#%s}{\\textsf{%s}}$" % (color, label)
    for label, color in labels
    if label in content_labels
]

# Generate strings describing added labels
title_labels_str = ""
if len(title_labels) == 0:
    title_labels_str = "I did not find any labels to add based on the title."
elif len(title_labels_new) != 0:
    arr_str = str(title_labels_new).strip("[]").replace("'", "")
    title_labels_str = (
        "I have added the following labels to this PR based on the title: "
        f"**[ {arr_str} ]**."
    )
    if len(title_labels) != len(title_labels_new):
        arr_str = (
            str(set(title_labels) - set(title_labels_new)).strip("[]").replace("'", "")
        )
        title_labels_str += (
            f" The following labels were already present: **[ {arr_str} ]**"
        )

content_labels_str = ""
if len(content_labels) != 0:
    if content_labels_status == "used":
        arr_str = str(content_labels).strip("[]").replace("'", "")
        content_labels_str = (
            "I have added the following labels to this PR based on "
            f"the changes made: **[ {arr_str} ]**. Feel free "
            "to change these if they do not properly represent the PR."
        )
    elif content_labels_status == "ignored":
        arr_str = str(content_labels).strip("[]").replace("'", "")
        content_labels_str = (
            "I would have added the following labels to this PR "
            f"based on the changes made: **[ {arr_str} ]**, "
            "however some package labels are already present."
        )
    elif content_labels_status == "large":
        content_labels_str = (
            "This PR changes too many different packages (>3) for "
            "automatic addition of labels, please manually add package "
            "labels if relevant."
        )
elif title_labels_str == "":
    content_labels_str = (
        "I did not find any labels to add that did not already "
        "exist. If the content of your PR changes, make sure to "
        "update the labels accordingly."
    )

# Create comment on pull request
pr.create_issue_comment(
    f"""
## Thank you for contributing to `tsml-py`

{title_labels_str}
{content_labels_str}

The [Checks](https://github.com/time-series-machine-learning/tsml-py/pull/{pr_number}/checks) tab will show the status of our automated tests. You can click on individual test runs in the tab or "Details" in the panel below to see more information if there is a failure.
    """
)
