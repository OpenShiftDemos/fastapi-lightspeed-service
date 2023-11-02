
from modules.task_breakdown import TaskBreakdown, DEFAULT_MODEL
import argparse

from modules.task_performer import TaskPerformer
from modules.task_processor import TaskProcessor
from modules.task_rephraser import TaskRephraser
from modules.yes_no_classifier import YesNoClassifier


def main(): 
    parser = argparse.ArgumentParser(
        description="light speed cli for task execution "
    )
    
    # common argument 
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        help="Set Verbose status of langchains [True/False]",
    )
    
    parser.add_argument(
        "-c",
        "--conversation-id",
        default="1234",
        type=str,
        help="A short identifier for the conversation",
    )
    
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, type=str, help="The model to use"
    )
    
    parser.add_argument(
        "-q",
        "--query",
        default="What is the weather like today?",
        type=str,
        help="The user query to use",
    )

    #     
    subparsers = parser.add_subparsers(dest="task")
    
    # tasks definitions 
    subparsers.add_parser("breakdown", help="Search the RAG and find a summary document with tasks")
    subparsers.add_parser("classifier", help="Process a list of tasks")

    
    perform_parser = subparsers.add_parser("perform", help="Perform a task")
    perform_parser.add_argument("perform_args", nargs="*", help="Arguments for the 'perform' task")
    
    
    processor_parser = subparsers.add_parser("processor", help="Process a list of tasks")
    processor_parser.add_argument("processor_args", nargs="*", help="Arguments for the 'processor' task")   
    
    rephraser_parser = subparsers.add_parser("rephraser", help="Rephrase a combination of a task and a user query into a single request")
    rephraser_parser.add_argument("rephraser_args", nargs="*", help="Arguments for the 'rephraser' task")   
    
    # unique arguments 
    perform_parser.add_argument(
        "-t",
        "--task",
        default="Create an autoscaler YAML that scales the cluster up to 10 nodes",
        type=str,
        help="A task to perform",
    )
    
    processor_parser.add_argument(
        "-t",
        "--tasklist",
        default="task1,task2",
        type=str,
        help="A comma-separated list of tasks to process eg: 'task1,task2'",
    )

    rephraser_parser.add_argument(
        "-t",
        "--task",
        default="Create an autoscaler YAML that scales the cluster up to 10 nodes",
        type=str,
        help="A task to perform",
    )


    # pass args to tasks 
    args = parser.parse_args()
    
    if args.task == "breakdown":        
        task_breakdown = TaskBreakdown()
        task_breakdown.breakdown_tasks(
            args.conversation_id, args.query, model=args.model, verbose=args.verbose
        )

    if args.task == "perform": 
        task_performer = TaskPerformer()
        task_performer.perform_task(
            args.conversation_id, args.perform_args.task, model=args.model, verbose=args.verbose
        )
        
    if args.task == "processor": 
            task_breakdown = TaskProcessor()
            task_breakdown.process_tasks(
                args.conversation_id,
                args.processor_args.tasklist.split(","),
                args.query,
                model=args.model,
                verbose=args.verbose,
            )
            
    if args.task == "rephraser": 
        task_rephraser = TaskRephraser()
        task_rephraser.rephrase_task(
            args.conversation_id,
            args.rephraser_args.task,
            args.query,
            model=args.model,
            verbose=args.verbose,
        )
        
    if args.task == "classifier":     
        yes_no_classifier = YesNoClassifier()
        yes_no_classifier.classify(
            args.conversation_id, args.query, model=args.model, verbose=args.verbose
        )

if __name__ == "__main__":
    main()
