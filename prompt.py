USER_PROMPT = """Question: {}\nOptions: {}"""  # simplified for formatting


SYSTEM_PROMPT_no_CoT = """Please select the correct answer from the options.

TASK REQUIREMENTS:
1. You MUST respond in the following format:
ANSWER: <A/B/C/D>"""

SYSTEM_PROMPT_CoT = """The following are multiple choice questions. Think step by step and then finish your answer.

TASK REQUIREMENTS:
1. Provide your thought process for arriving at the answer.
2. You MUST respond in the following format:
REASON: <The justification for your answer>
ANSWER: <A/B/C/D>
"""

# SYSTEM_PROMPT_no_CoT = """
# You are given a multiple choice question with an image. 
# Please select the correct answer from the options provided.

# TASK REQUIREMENTS:
# 1. Respond ONLY in the following format:
# ANSWER: <A/B/C/D>

# DO NOT explain your reasoning. DO NOT provide extra text.
# """

# SYSTEM_PROMPT_CoT = """
# You are given a multiple choice question with an image. 
# Think step by step before selecting the answer.

# TASK REQUIREMENTS:
# 1. Explain your reasoning process.
# 2. Then conclude with the following format:
# REASON: <Your justification>
# ANSWER: <A/B/C/D>

# Any answer not following this format will be considered invalid.
# """
