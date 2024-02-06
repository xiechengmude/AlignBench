from inference.models import api_model
from openai import OpenAI
import json

class xdan_v6(api_model):
    def __init__(self, workers=10):
        self.workers = workers
        self.model_name = "xDAN-L2-Chat-RL-v6.1-v2-0203-e2" # TODO
        self.temperature = 0.7
        self.system_prompt = """You are a helpful assistant named xDAN and to be an expert in worldly knowledge, skilled in employing a probing questioning strategy, 
carefully considering each step before providing answers.Take a deep breath and find out the principles behind this questions.
Then FOLLOWING the BELOW RULES , REPHASE the question and apply the principle to response directly."""

    def get_api_result(self, sample):
        question = sample["question"]
        temperature = sample.get("temperature", self.temperature)

        def single_turn_wrapper(text):
            return [{"role": "system", "content": self.system_prompt}
            ,{"role": "user", "content": text}]

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=single_turn_wrapper(question),
            temperature=temperature
        )

        output = response.choices[0].message.content
        return output