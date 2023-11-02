import logging
import sys
from string import Template
from modules.model_context import get_watsonx_predictor

DEFAULT_MODEL = "ibm/granite-13b-chat-grounded-v01"


class TaskPerformer:
    def __init__(self):
        logging.basicConfig(
            stream=sys.stdout,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("task_rephraser")

    def perform_task(self, conversation, task, **kwargs):
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
            '{"conversation": "$conversation", "task": "$task", "model": "$model", "verbose": "$verbose"}'
        )

        self.logger.info(
            conversation
            + " call settings: "
            + settings_string.substitute(
                conversation=conversation, task=task, model=model, verbose=verbose
            )
        )

        # determine if this should go to a general LLM, the YAML generator, or elsewhere

        # send to the right tool

        # output the response
        response = """
apiVersion: "autoscaling.openshift.io/v1"
kind: "ClusterAutoscaler"
metadata:
  name: "default"
spec:
  resourceLimits:
    maxNodesTotal: 10
  scaleDown: 
    enabled: true 
"""

        self.logger.info(conversation + " response: " + response)

        # return the response

        return response