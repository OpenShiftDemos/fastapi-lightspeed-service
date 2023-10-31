import logging
import sys
from modules.model_context import get_watsonx_predictor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from string import Template

DEFAULT_MODEL = "ibm/granite-13b-instruct-v1"


class QuestionValidator:
    def __init__(self):
        logging.basicConfig(
            stream=sys.stdout,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("question_validator")

    def validate_question(self, conversation, query, **kwargs):
        if "model" in kwargs:
            model = kwargs["model"]
        else:
            model = DEFAULT_MODEL

        if "verbose" in kwargs:
            if kwargs["verbose"] == "True" or kwargs["verbose"] == "true":
                verbose = True
            else:
                verbose = False
        else:
            verbose = False

        # TODO: must be a smarter way to do this
        settings_string = Template(
            '{"conversation": "$conversation", "query": "$query","model": "$model", "verbose": "$verbose"}'
        )

        self.logger.info(
            conversation
            + " call settings: "
            + settings_string.substitute(
                conversation=conversation,
                query=query,
                model=model,
                verbose=verbose,
            )
        )

        prompt_instructions = PromptTemplate.from_template(
            """
Instructions:
- You are a question classifying tool
- You are an expert in kubernetes and openshift
- Your job is to determine if a question is about kubernetes or openshift and to provide a one word response
- If a question is not about kubernetes or openshift, answer with only the word "INVALID"
- If a question is about kubernetes or openshift, answer with only the word "VALID"
- Do not provide explanation, only respond with the single chosen word

Example Question:
Can you make me lunch with ham and cheese?
Example Response:
INVALID

Example Question:
Why is the sky blue?
Example Response:
INVALID

Example Question:
Can you help configure my cluster to automatically scale?
Example Response:
VALID

Question:
{query}
Response:
"""
        )

        self.logger.info(conversation + " Validatin query")
        self.logger.info(conversation + " usng model: " + model)

        bare_llm = get_watsonx_predictor(model=model, min_new_tokens=1, max_new_tokens=2)
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        task_query = prompt_instructions.format(query=query)

        self.logger.info(conversation + " task query: " + task_query)

        response = llm_chain(inputs={"query": query})

        self.logger.info(conversation + " response: " + str(response))
        return response['text']

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate whether or not a question is about Kubernetes and OpenShift"
    )
    parser.add_argument(
        "-c",
        "--conversation-id",
        default="1234",
        type=str,
        help="A short identifier for the conversation",
    )
    parser.add_argument(
        "-q",
        "--query",
        default="Can you make me lunch with ham and cheese?",
        type=str,
        help="The user query to use",
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, type=str, help="The model to use"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        help="Set Verbose status of langchains [True/False]",
    )

    args = parser.parse_args()

    question_validator = QuestionValidator()
    question_validator.validate_question(
        args.conversation_id,
        args.query,
        model=args.model,
        verbose=args.verbose,
    )

