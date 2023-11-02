import logging
import sys
from modules.model_context import get_watsonx_predictor
from modules.yes_no_classifier import YesNoClassifier
from modules.task_performer import TaskPerformer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from string import Template

DEFAULT_MODEL = "ibm/granite-13b-chat-grounded-v01"


class TaskRephraser:
    def __init__(self):
        logging.basicConfig(
            stream=sys.stdout,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("task_rephraser")

    def rephrase_task(self, conversation, task, original_query, **kwargs):
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
            '{"conversation": "$conversation", "task": "$task", "query": "$original_query","model": "$model", "verbose": "$verbose"}'
        )

        self.logger.info(
            conversation
            + " call settings: "
            + settings_string.substitute(
                conversation=conversation,
                task=task,
                original_query=original_query,
                model=model,
                verbose=verbose,
            )
        )

        prompt_instructions = PromptTemplate.from_template(
            """
            Instructions:
            - You are a helpful assistant.
            - Your job is to combine the information from the task and query into a single, new task.
            - Base your answer on the provided task and query and not on prior knowledge.

            TASK:
            {task}
            QUERY:
            {query}

            Please combine the information from the task and query into a single, new task.

            Response:
            """
        )

        self.logger.info(conversation + " Rephrasing task and query")
        self.logger.info(conversation + " usng model: " + model)

        bare_llm = get_watsonx_predictor(model=model, min_new_tokens=5)
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        task_query = prompt_instructions.format(task=task, query=original_query)

        self.logger.info(conversation + " task query: " + task_query)

        response = llm_chain(inputs={"task": task, "query": original_query})

        self.logger.info(conversation + " response: " + str(response))
        return response['text']
