from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from MCP_Client import setup
from MCP_Client.prompt_template import *
from MCP_Client.client_utils import *

class ChatRequest(BaseModel):
    user_query: str

class ImageRequest(BaseModel):
    base64_img: str

# MCP Variables
session, flagging_tools, routing_tools, editing_tools = None, None, None, None
img_index = 1

@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, flagging_tools, routing_tools, editing_tools
    # Open MCP connection + tools
    session, flagging_tools, routing_tools, editing_tools = await setup.bootstrap()
    try:
        yield
    finally:
        await setup.shutdown()

app = FastAPI(lifespan=lifespan)

@app.post("/sent-two-images")
async def chat_inference_two_images(req: ChatRequest, img1: ImageRequest, img2: ImageRequest):
    global session, img_index
    # Read current user query
    current_user_query = req.user_query
    # Read query image 
    query_image = base64_to_pil(img1.base64_img)
    ref_image = base64_to_pil(img2.base64_img)
    url_query_image = base64_conversion(query_image) 
    print("QUERY IMAGE - RETRIEVED")
    # Restart History
    img_index = 1
    with open(f"{session_path}/history.json", "w") as f:
        json.dump({"image_initial_query":url_query_image}, f, indent=4)
    # OS Makedirs
    os.makedirs(f"{session_path}/temp", exist_ok=True)
    os.makedirs(f"{session_path}/stored_images", exist_ok=True)
    # Save Images
    query_image.save(f"{session_path}/stored_images/query.png")
    query_image.save(f"{session_path}/temp/result.png")
    ref_image.save(f"{session_path}/temp/ref.png")
    # Define user prompt calling
    user_prompt_calling = f"""
    SESSION PATH : {session_path}
    USER QUERY : {current_user_query}"""
    # Generate Tool Response
    url_result_image = asyncio.run(executing_function_calling_image_editing(user_prompt_calling, url_query_image, editing_tools, session))
    print("FUNCTION CALLING IMAGE EDTING - ENDED")
    # Save the Result
    base64_to_pil(url_result_image).save(f"{session_path}/stored_images/response_{img_index}.png")
    img_index += 1
    # Save the query and response
    save_chat_session(current_user_query, url_result_image)
    print("CHAT SASSION - SAVED")
    return {"response":extract_base64(url_result_image)}

@app.post("/sent-one-image")
async def chat_inference_one_image(req: ChatRequest, img: ImageRequest):
    global session, img_index
    # Read current user query
    current_user_query = req.user_query
    # Reference indication checking
    reference_flag = asyncio.run(get_reference_flagging(current_user_query, flagging_tools))
    # Check condition
    if reference_flag==0: 
        # Read Query Image
        query_image = base64_to_pil(img.base64_img)
        url_query_image = base64_conversion(query_image) 
        # Restart History
        img_index = 1
        with open(f"{session_path}/history.json", "w") as f:
            json.dump({"image_initial_query":url_query_image}, f, indent=4)
        # OS Makedirs
        os.makedirs(f"{session_path}/temp", exist_ok=True)
        os.makedirs(f"{session_path}/stored_images", exist_ok=True)
        # Save Image
        base64_to_pil(url_query_image).save(f"{session_path}/stored_images/query.png")
        base64_to_pil(url_query_image).save(f"{session_path}/temp/result.png")
    else: # Find query image through history
        # Read ref Image and save it
        base64_to_pil(img.base64_img).save(f"{session_path}/temp/ref.png")
        # Open Session history and construct history payload
        history_payload = extract_history_payload(current_user_query)
        # Get ref number of query image given history payload
        ref_number = asyncio.run(get_query_image_ref_number(history_payload, routing_tools))
        # Get Corresponding query image
        with open(f"{session_path}/history.json","r") as f:
            data = json.load(f)
        if ref_number == 0:
            url_query_image = data["image_initial_query"]
        else:
            url_query_image = data["history"][f"{ref_number}"]["image_response"]
        # Save 
        base64_to_pil(url_query_image).save(f"{session_path}/temp/result.png")
    print("QUERY IMAGE - RETRIEVED")
    # Define user prompt calling
    user_prompt_calling = f"""
    SESSION PATH : {session_path}
    USER QUERY : {current_user_query}"""
    # Generate Tool Response
    url_result_image = asyncio.run(executing_function_calling_image_editing(user_prompt_calling, url_query_image, editing_tools, session))
    print("FUNCTION CALLING IMAGE EDTING- ENDED")
    # Save the Result
    base64_to_pil(url_result_image).save(f"{session_path}/stored_images/response_{img_index}.png")
    img_index += 1
    # Save the query and response
    save_chat_session(current_user_query, url_result_image)
    print("CHAT SASSION - SAVED")
    return {"response":extract_base64(url_result_image)}
    
@app.post("/sent-only-text-query")
async def chat_inference_only_text_query(req: ChatRequest):
    global session, img_index
    # Read current user query
    current_user_query = req.user_query
    # Open Session history and construct history payload
    print("CHECKING HISTORY")
    history_payload = extract_history_payload(current_user_query)
    # Get ref number of query image given history payload
    ref_number = asyncio.run(get_query_image_ref_number(history_payload, routing_tools))
    # Get Corresponding query image
    with open(f"{session_path}/history.json","r") as f:
        data = json.load(f)
    if ref_number == 0:
        url_query_image = data["image_initial_query"]
    else:
        url_query_image = data["history"][f"{ref_number}"]["image_response"]
    # Save 
    base64_to_pil(url_query_image).save(f"{session_path}/temp/result.png")
    print("QUERY IMAGE - RETRIEVED")
    # Define user prompt calling
    user_prompt_calling = f"""
    SESSION PATH : {session_path}
    USER QUERY : {current_user_query}"""
    # Generate Tool Response
    url_result_image = await executing_function_calling_image_editing(user_prompt_calling, url_query_image, editing_tools, session)
    print("FUNCTION CALLING IMAGE EDTING- ENDED")
    # Save the Result
    base64_to_pil(url_result_image).save(f"{session_path}/stored_images/response_{img_index}.png")
    img_index += 1
    # Save the query and response
    save_chat_session(current_user_query, url_result_image)
    print("CHAT SASSION - SAVED")
    return {"response":extract_base64(url_result_image)}
