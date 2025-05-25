import streamlit as st
import pulp
import json
import openai
import time
import os
from dotenv import load_dotenv
from pulp import LpContinuous, LpInteger

########################################
# Build LP Problem from User Inputs
########################################
def build_lp_problem(lp_type, objective_str, variables_text, constraints_text):
    if lp_type.lower() == "maximize":
        prob = pulp.LpProblem("UserLP", pulp.LpMaximize)
    else:
        prob = pulp.LpProblem("UserLP", pulp.LpMinimize)

    var_dict = {}
    for line in variables_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            st.error(f"Line '{line}' must include at least a variable name and a lower bound.")
            continue
        var_name = parts[0].strip()
        try:
            lower_bound = float(parts[1].strip())
        except Exception as e:
            st.error(f"Error parsing lower bound for {var_name}: {e}")
            continue
        if len(parts) == 3:
            try:
                upper_bound = float(parts[2].strip())
            except Exception as e:
                st.error(f"Error parsing upper bound for {var_name}: {e}")
                continue
            var_dict[var_name] = pulp.LpVariable(var_name, lowBound=lower_bound, upBound=upper_bound)
        else:
            var_dict[var_name] = pulp.LpVariable(var_name, lowBound=lower_bound)
       
    try:
        obj_expr = eval(objective_str, {}, var_dict)
    except Exception as e:
        st.error(f"Error evaluating objective expression: {e}")
        return None
    prob += obj_expr, "Objective"

    for line in constraints_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "<=" in line:
            parts = line.split("<=")
            op = "<="
        elif ">=" in line:
            parts = line.split(">=")
            op = ">="
        elif "<" in line:
            parts = line.split("<")
            op = "<"
        elif ">" in line:
            parts = line.split(">")
            op = ">"
        else:
            st.error(f"Constraint not recognized (no inequality): {line}")
            continue
        if len(parts) != 2:
            st.error(f"Invalid constraint format: {line}")
            continue
        left_expr_str = parts[0].strip()
        right_expr_str = parts[1].strip()
        try:
            left_expr = eval(left_expr_str, {}, var_dict)
        except Exception as e:
            st.error(f"Error evaluating left side of constraint '{line}': {e}")
            continue
        try:
            right_value = float(right_expr_str)
        except Exception as e:
            st.error(f"Error parsing right side of constraint '{line}': {e}")
            continue
        if op == "<=":
            prob += (left_expr <= right_value)
        elif op == ">=":
            prob += (left_expr >= right_value)
        elif op == "<":
            prob += (left_expr < right_value)
        elif op == ">":
            prob += (left_expr > right_value)

    return prob

########################################
# Solve the LP Problem and Return Results
########################################
def solve_problem(prob):
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    var_values = {var.name: var.varValue for var in prob.variables()}
    shadow_prices = {name: constraint.pi for name, constraint in prob.constraints.items()}
    reduced_costs = {var.name: var.dj for var in prob.variables()}
    results = {
        "status": status,
        "objective": obj_val,
        "variables": var_values,
        "shadow_prices": shadow_prices,
        "reduced_costs": reduced_costs
    }
    return results


    
def get_lp_answer(lp_results_json: str, user_question: str, business_context: str = "") -> str:
    lp_results = json.loads(lp_results_json)
    prompt = f"""
Business context:
{business_context or "No additional context provided."}

Here are the LP results:
- Status: {lp_results.get("status")}
- Variables: {lp_results.get("variables")}
- Objective: {lp_results.get("objective")}
- Shadow Prices: {lp_results.get("shadow_prices")}
- Reduced Costs: {lp_results.get("reduced_costs")}

User question: {user_question}
"""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Your task is to interpret linear programming (LP) results and provide recommendations in a way that a non-technical manager can easily understand. Focus on practical business insights (profitability, resource use, bottlenecks, trade-offs). Explain economic reasoning clearly (e.g., how resource changes affect profit). Avoid complex mathematical terms unless absolutely necessary; if you must use one, explain it simply. Structure your answers logically, using short paragraphs or bullet points if helpful. Maintain a professional but accessible tone — imagine you are briefing an operations director or supply chain manager. Summarize your main point clearly at the end of each response if possible. If asked -what if- or -scenario- questions, clearly explain cause-effect relationships. If context is updated in the conversation (e.g., a constraint changes), adapt your explanation accordingly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response["choices"][0]["message"]["content"]


def get_confidence_level(lp_results_json: str, user_question: str) -> str:
    prompt = f"""
Based on the following LP results and user question, assess the confidence level of answering it using linear programming principles.

LP results (shortened): {lp_results_json[:1000]}
User question: {user_question}

Respond with one word only: "low", "medium", or "high".
"""

    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a confidence-level estimator for LP model questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response["choices"][0]["message"]["content"].strip().lower()



########################################
# GPT-Assisted LP Generator
########################################
def generate_lp_from_natural_language(natural_text: str):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    user_prompt = f"""
Given the following business problem, create a JSON with keys:
- "objective": a string defining the objective expression, starting with 'maximize' or 'minimize'
- "variables": a list of variable strings in the format "name,lower_bound[,upper_bound]"
- "constraints": a list of constraint strings like "xA + 3*xB <= 200"
- "business_context": a plain language description of what each variable and constraint represents

Business problem description:
{natural_text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a linear programming assistant."},
                  {"role": "user", "content": user_prompt}],
        temperature=0.3
    )
    content = response["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Could not parse GPT response as JSON.", "raw": content}

########################################
# Main App
########################################
def main():
    st.set_page_config(page_title="LP + GPT-4 Chatbot", layout="wide")
    st.title("LP Assistant")

    # Session state setup
    if "lp_json" not in st.session_state:
        st.session_state["lp_json"] = ""
    if "lp_definition" not in st.session_state:
        st.session_state["lp_definition"] = {}
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = 0

    tabs = st.tabs(["LP Builder", "Chatbot", "AI LP Generator"])

    ########################
    # LP Builder Tab
    ########################
    with tabs[0]:
        st.header("Define and Solve your LP")
        lp_type = st.radio("LP Type", ["Maximize", "Minimize"],
                           index=["Maximize", "Minimize"].index(st.session_state.get("lp_type", "Maximize")))
        
        # Objective function
        st.markdown("**Objective Function**")
        st.markdown("Example:")
        st.code("30*xA + 50*xB", language="text")
        objective_str = st.text_input("Enter your objective function:", value=st.session_state.get("objective_str", ""))
        
        # Variables
        st.markdown("**Variables**")
        st.markdown("Provide them one per line (variable, lower_bound,[upper_bound]).")
        st.markdown("Example:")
        st.code("xA,20\nxB,10,100", language="text")
        variables_text = st.text_area("Enter your variables:", value=st.session_state.get("variables_text", "")) 
    
        # Constraints
        st.markdown("**Constraints**")
        st.markdown("Provide them one per line, using <=, >=, or =.")
        st.markdown("Example:")
        st.code("xA + xB <= 100\nxA + 3*xB <= 200", language="text")
        constraints_text = st.text_area("Enter your constraints:", value=st.session_state.get("constraints_text", ""))
        
        # Business context
        st.markdown("**Business Context**")
        st.text_area("Describe your business problem:", key="business_context", height=150) #attention to key: StreamlitAPIException

        if st.button("Build and Solve LP"):
            prob = build_lp_problem(lp_type, objective_str, variables_text, constraints_text)
            if prob:
                results = solve_problem(prob)
                st.session_state["lp_json"] = json.dumps(results)
                st.session_state["lp_definition"] = {
                    "lp_type": lp_type,
                    "objective": objective_str,
                    "variables": variables_text,
                    "constraints": constraints_text,
                    "business_context": st.session_state.get("business_context", "")
                }
                st.success("LP solved. Switch to the Chatbot tab to ask questions.")
                st.write("**Results:**")
                st.write(results)

    ########################
    # Chatbot Tab
    ########################
    with tabs[1]:
        st.header("Ask GPT-4 About Your LP")
        if not st.session_state["lp_json"]:
            st.info("Solve a model first in the LP Builder tab.")
        else:
            example_questions = [
                "What does the objective value tell us?",
                "Which resource is the bottleneck?",
                "If I could increase one constraint, which would help the most?",
                "Are any variables not contributing to the objective?",
                "What happens if I increase the budget by 10 units?"
            ]

            st.markdown("### 💡 Example questions")
            cols = st.columns(len(example_questions))
            for i, question in enumerate(example_questions):
                if cols[i].button(question):
                    st.session_state["messages"].append({"role": "user", "content": question})
                    start_time = time.time()

                    reply = get_lp_answer(
                        st.session_state["lp_json"],
                        question,
                        st.session_state["lp_definition"].get("business_context", "")
                    )
                    confidence = get_confidence_level(
                        st.session_state["lp_json"],
                        question
                    )
                    elapsed = time.time() - start_time

                    st.session_state["messages"].append({"role": "assistant", "content": reply})
                    st.session_state["messages"].append({"role": "caption", "content": f"🔒 Confidence: {confidence}"})
                    st.session_state["messages"].append({"role": "caption", "content": f"🕒 Response time: {elapsed:.2f} seconds"})
                    st.rerun()

            st.divider()

            for msg in st.session_state["messages"]:
                if msg["role"] == "caption":
                    st.caption(msg["content"])
                else:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

            user_input = st.chat_input("Ask a question...")
            if user_input:
                st.session_state["messages"].append({"role": "user", "content": user_input})
                start_time = time.time()

                reply = get_lp_answer(
                    st.session_state["lp_json"],
                    user_input,
                    st.session_state["lp_definition"].get("business_context", "")
                )
                confidence = get_confidence_level(
                    st.session_state["lp_json"],
                    user_input
                )
                elapsed = time.time() - start_time

                st.session_state["messages"].append({"role": "assistant", "content": reply})
                st.session_state["messages"].append({"role": "caption", "content": f"🔒 Confidence: {confidence}"})
                st.session_state["messages"].append({"role": "caption", "content": f"🕒 Response time: {elapsed:.2f} seconds"})
                st.rerun()

    ########################
    # AI LP Generator Tab
    ########################
    with tabs[2]:
        st.header("Generate LP from Natural Language")
        natural_text = st.text_area("Describe your business problem in plain language")
        if st.button("Generate LP Model"):
            result = generate_lp_from_natural_language(natural_text)
            if "error" in result:
                    st.error(result["error"])
                    st.code(result["raw"])
            else:
                st.session_state["lp_generated_result"] = result
                st.success("Model generated")        


        if "lp_generated_result" in st.session_state:
                result = st.session_state["lp_generated_result"]

                with st.form("edit_and_use_form"):
                    st.subheader("Edit Before Using")
                    edited_objective = st.text_input("Objective Function", result["objective"].split(" ", 1)[1])
                    edited_variables = st.text_area("Variables", "\n".join(result["variables"]))
                    edited_constraints = st.text_area("Constraints", "\n".join(result["constraints"]))
                    edited_context = st.text_area("Business Context", result["business_context"], height=150)

                    use_now = st.form_submit_button("Use This Model")
                
                if use_now:
                    if "business_context" not in st.session_state:
                        st.session_state["business_context"] = edited_context
                    st.session_state["lp_type"] = "Maximize" if result["objective"].lower().startswith("maximize") else "Minimize"
                    st.session_state["objective_str"] = edited_objective
                    st.session_state["variables_text"] = edited_variables
                    st.session_state["constraints_text"] = edited_constraints
                    st.session_state["active_tab"] = 0

                    st.info("Go to the LP Builder tab to solve the model.")
                    #st.experimental_rerun()
        else:
            st.info("Enter a business description above and click 'Generate LP Model'.")



if __name__ == "__main__":
    main()
