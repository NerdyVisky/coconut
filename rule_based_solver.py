import json
import re
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_expression(expression: str, computed_values: Dict[str, float]) -> float:
    """Evaluate the LHS expression while substituting known computed values using BODMAS."""
    
    # Substitute previously computed values
    for key, value in computed_values.items():
        expression = expression.replace(key, str(value))
    
    try:
        return float(eval(expression, {"__builtins__": None}, {}))  # Safe evaluation
    except Exception:
        # Manual computation attempt using BODMAS
        try:
            return float(eval(re.sub(r'(?<=\d)(?=[-+*/()])|(?<=[-+*/()])(?=\d)', ' ', expression)))
        except Exception:
            tokens = re.findall(r'\d+\.?\d*|[-+*/()]', expression)
            values = []
            operators = []
            precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
            
            def apply_operator():
                b = values.pop()
                a = values.pop()
                op = operators.pop()
                if op == '+':
                    values.append(a + b)
                elif op == '-':
                    values.append(a - b)
                elif op == '*':
                    values.append(a * b)
                elif op == '/':
                    values.append(a / b if b != 0 else float('inf'))
            
            i = 0
            try:
                while i < len(tokens):
                    token = tokens[i]
                    if token.isnumeric() or '.' in token:
                        values.append(float(token))
                    elif token in precedence:
                        while (operators and operators[-1] in precedence and
                            precedence[operators[-1]] >= precedence[token]):
                            apply_operator()
                        operators.append(token)
                    elif token == '(':
                        operators.append(token)
                    elif token == ')':
                        while operators and operators[-1] != '(':
                            apply_operator()
                        operators.pop()
                    i += 1
            except: 
                print("Error evaluting" + expression)
                raise ValueError("ERR!")
            
            while operators:
                apply_operator()
            
            return float(values[0]) if values else float(tokens[0])
# def evaluate_expression(lhs: str, computed_values: Dict[str, float]) -> float:
#     """Evaluate the LHS expression while substituting known computed values."""
#     tokens = re.split(r'(\+|\-|\*|/)', lhs)
#     operands = [t.strip() for t in tokens if t.strip() and t.strip() not in '+-*/']
#     operator = next((t.strip() for t in tokens if t.strip() in '+-*/'), None)
    
#     # Substitute previously computed values
#     for i, op in enumerate(operands):
#         if op in computed_values:
#             operands[i] = str(computed_values[op])
    
#     # Compute the result using floating point arithmetic
#     try:
#         if operator == '+':
#             return float(operands[0]) + float(operands[1])
#         elif operator == '-':
#             return float(operands[0]) - float(operands[1])
#         elif operator == '*':
#             return float(operands[0]) * float(operands[1])
#         elif operator == '/':
#             return float(operands[0]) / float(operands[1])  # Floating point division
#         else:
#             print(lhs + " not a identified expression, returning first operand")
#             return float(operands[0])
#     except:
#         print(lhs)
#         return float(1)
#         # raise ValueError("Unknown operator in expression")

def process_example(example: Dict) -> Dict:
    """Process a single example to fix calculation errors using rule-based solving."""
    steps = re.findall(r'<<([^=]+)=([^>>]+)>>', example["text_output"])
    computed_values = {}
    computed_rhs = example["answer_output"]
    
    for lhs, rhs in steps:
        lhs = lhs.strip()
        rhs = rhs.strip()
        computed_rhs = evaluate_expression(lhs, computed_values)
        computed_values[rhs] = computed_rhs
    
    example["answer_output_tool"] = str(computed_rhs)  # Store the final computed result
    return example

def rule_based_solver(data: List[Dict]) -> List[Dict]:
    """Apply the rule-based solver to a list of examples."""
    return [process_example(example) for example in data]

def compute_statistics(data: List[Dict]):
    """Compute accuracy statistics and plot heatmap."""
    original_correct = sum(1 for x in data if x["Original_Result"])
    tool_correct = sum(1 for x in data if x["With_tool_result"])
    total = len(data)
    
    print(f"Accuracy without rule-based tool: {original_correct / total:.2%}")
    print(f"Accuracy with rule-based tool: {tool_correct / total:.2%}")
    
    heatmap_data = {
        "True-True": sum(1 for x in data if x["Original_Result"] and x["With_tool_result"]),
        "False-False": sum(1 for x in data if not x["Original_Result"] and not x["With_tool_result"]),
        "False-True": sum(1 for x in data if not x["Original_Result"] and x["With_tool_result"]),
        "True-False": sum(1 for x in data if x["Original_Result"] and not x["With_tool_result"]),
    }
    
    plt.figure(figsize=(5, 5))
    sns.heatmap([[heatmap_data["True-True"], heatmap_data["True-False"]],
                 [heatmap_data["False-True"], heatmap_data["False-False"]]],
                annot=True, fmt="d", cmap="Blues",
                xticklabels=["Tool Correct", "Tool Incorrect"],
                yticklabels=["Original Correct", "Original Incorrect"])
    plt.xlabel("Rule-Based Tool")
    plt.ylabel("Original Computation")
    plt.title("Accuracy Heatmap")
    plt.show()


def compute_accuracy(data: List[Dict]) -> List[Dict]:
    """Compute accuracy by comparing expected and computed answers with tolerance of ±1 integer value."""
    for example in data:
        try:
            answer = int(float(example["answer"]))
            original_result = int(float(example["answer_output"]))
            with_tool_result = int(float(example["answer_output_tool"]))
            
            example["Original_Result"] = abs(answer - original_result) <= 1
            example["With_tool_result"] = abs(answer - with_tool_result) <= 1
        except Exception:
            example["Original_Result"] = False
            example["With_tool_result"] = False
    
    return data


# Example usage
if __name__ == "__main__":
    with open("/Users/nerdyvisky/Documents/ETNLP_Project/raw_outputs/raw_output_log_gpt2_cot.json", "r") as f:
        data = json.load(f)
    
    processed_data = rule_based_solver(data)
    processed_data = compute_accuracy(processed_data)
    compute_statistics(processed_data)
    
    with open("/Users/nerdyvisky/Documents/ETNLP_Project/raw_outputs/raw_output_log_gpt2_cot_rulebased.json", "w") as f:
        json.dump(processed_data, f, indent=4)
        
    
