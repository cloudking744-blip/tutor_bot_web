import ollama
from ollama import AsyncClient , Client
from fastapi import APIRouter, Request
router = APIRouter()

api_key = "b7ed6388e74641688a490d04609e53a5.qFoiRU9TWID2wL_mDCD-cJrN"
client = AsyncClient(host = "https://ollama.com",
                    headers={"Authorization": f"Bearer {api_key}"})


def model_list():
    models = ollama.list()

    models = [model.model for model in models.models]

    return {"models": models}

async def message( model_name, tokens, messages):
    tokens = tokens
    ollama_messages = [{"role": "system",
                      "content":"""
                        1.following is your name , character and abilities definition
                            name: Riya
                            sex : female
                            character : you are a intelligent coding and tutor assistant in the domain
                                        1.react
                                        2.python
                                        3.dl learning model development
                                        4.mathematics tutor
                                        5.robotics tutor and developer
                                        6.control engineering 
                                        7.swarm robotics etc  
                                        8.code debugger with approaches like console log or print function to check and debug the code at various stages when user asks
                                        9.generic critic of plans that user presents and asks for constructive criticism
                                        10.You are also a english and formal communication assistant , who fixes the langauge into formal and buisness english for communication 
                                        11.you can use html code for visualisation of following 
                                            11.1.Graphs 
                                            11.2.Interactive graphs
                                            11.3.Interactive Graphical Visualisation of concepts
                                        12.Dots(.) , Dashes (-) , Less then / greater then (< >) ,star(*) , minus ( - ) and brackets can also be used to show drawing or visualization
                                        13.Dont insert color or other css tags etc in codes tag or answers 
                                        14.Dont use garbage value like (&nbsp;) etc
                                        15. use symbols where ever required in answering instead of using words for the symbol like use α instead of alpha 
                        """,
                      }]
    ollama_messages.extend(messages)
    print(f"ollama_messages received {ollama_messages}")
    if model_name == "gpt-oss:120b-cloud":
        raw_response = await client.chat(
            model=model_name,
            messages=ollama_messages,
            options={
                "num_predict": tokens
            }
        )
        answer = raw_response["message"]["content"]
        answer = await summarizer(answer)
        answer = await formator(answer)
        return answer

    else:
        raw_response = await client.chat(
        model=model_name,
        messages=ollama_messages,
        options={
            "num_predict": tokens
        }
    )
        answer = raw_response["message"]["content"]
        print(f"answer received {answer}")
        answer = await formator(answer)
        print(f"formated answer received {answer}")
        return answer



async def summarizer(answer_input):
    ollama_message = [
        {"role": "system",
        "content": """
                            1.following is your name , character and abilities definition
                                name: Riya
                                sex : female
                                character : you are a intelligent coding and tutor assistant in the domain
                                            1.react
                                            2.python
                                            3.dl learning model development
                                            4.mathematics tutor
                                            5.robotics tutor and developer
                                            6.control engineering 
                                            7.swarm robotics etc  
                                            8.code debugger with approaches like console log or print function to check and debug the code at various stages when user asks
                                            9.generic critic of plans that user presents and asks for constructive criticism
                                            10.You are also a english and formal communication assistant , who fixes the langauge into formal and buisness english for communication 
                                            11.you can use html code for visualisation of following 
                                                11.1.Graphs 
                                                11.2.Interactive graphs
                                                11.3.Interactive Graphical Visualisation of concepts
                                            12.Dots(.) , Dashes (-) , Less then / greater then (< >) ,star(*) , minus ( - ) and brackets can also be used to show drawing or visualization
                                            13.Dont insert color or other css tags etc in codes tag or answers 
                                            14.Dont use garbage value like (&nbsp;) etc
                                            15.you are content formator who re writes the given information full technical content in a better way """},
        {"role":"user",
         "content":answer_input},
     ]
    raw_response = await client.chat(
        model="gemma4:cloud",
        messages=ollama_message,
    )
    answer = raw_response["message"]["content"]
    return answer




async def formator(input_data):
    message = [
        {"role": "system",
         "content": """ 
         1.you are a input to output formator who get the input data and returns into the reformated html output
         with following condition 
         output format :
                            1.the output should be formatted in html tags 
                                1.1. like heading , paragraph or tables etc 
                                1.2. there are few additional tags i.e.
                                    1.2.1 <python> </python> for python code
                                    1.2.2 <react> </react> for react code 
                                    1.2.3. and so on code tags for any code in the output
                                    1.2.4. you can use these tags for the mathematical solutions or solution or part of message which should /can be copied
                                1.3. instead of dollar sign or $ sign you should use html tags for all practical purposes
                                1.4. there is no maths tag here
                                1.5. represent mathematical equation in exact text form rather then representing them in the form of tags
                                1.6  
                            2.If input already contains html code for visualization then pack in the output
                            3.the output should be only reformated data there should be not markdown or suggestions etc
                            4.Dont insert color or other css tags etc in codes tag  
                            """
         },
        {
            "role": "user",
            "content": input_data
        }
    ]
    raw_response = await client.chat(
        model="gemma4:31b-cloud",
        messages=message,
    )
    answer = raw_response["message"]["content"]
    if answer.startswith("```"):
        answer = answer.replace("```", "")
    elif answer.endswith("```"):
        answer = answer.replace("```", "")
    return answer



@router.post("/chat")
async def chatbot(request: Request):
    try:
        body = await request.json()

        messages = body["messages"]
        model_name = body["model"]
        tokens = body["options"]["num_predict"]
        print(f"tokens in body {tokens}")
        answer = await message(model_name, tokens, messages)
        return {"message":
                    {
                        "role":"assistant",
                        "content": answer
                    }}
    except Exception as e :
        return {"message":
            {
                "role": "assistant",
                "content":  e
            }}

@router.get("/model_list")
async def models(request: Request):
    model_lis = model_list()
    print(f"{model_lis}")
    return model_lis
