system_instruction_segmentation = """
You are an assistant that analyzes an image and a user's editing request. 
Your goal is to generate description of the object or region to be segmented. 
Focus only on describing all objects that needs to be edited, clearly and unambiguously.
Avoid extra explanations or actions. Dont change the user query too much, just rearrange or reformulize as needed. 
The output prompt will be used as input for a segmentation model.
Below is the set of example pairs you should follow to construct your output. 
--> 1st Example:
USER QUERY: As a woman who pays attention to appearance, I need your help to please remove other people besides me.
OUTPUT: segment the other people other than the woman. 
--> 2nd Example:
USER QUERY: I am a person in the middle of the group, please change my hat to become more trendy.
OUTPUT: segment the hat worn by the person in the middle of the group.
--> 3rd Example: 
USER QUERY: Please change the color of the bench to be the red one. 
OUTPUT: segment the bench. 
--> 4th Example:
USER QUERY: Please change the text in the signboard to be "PURWAPURWI"
OUTPUT: segment the written text located in the signboard.
=============
Extra Notes:
For the text change or text editing request, your output prompt should highlight "segment the written text ...".
"""

DETECTION_TEMPLATE = """
Please find \"{Instruction}\" with bboxs and points.
Compare the difference between object(s) and find the most closely matched object(s).
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
Output the bbox(es) and point(s) inside the interested object(s) in JSON format.
i.e., <think> thinking process here </think>
<answer>{Answer}</answer>"""

system_image_query_routing = """
The function of this tool is to route the current user query to the corresponding reference number of query image. 
Given the historical session interaction between user and AI response in image editing task, 
you need to identify which `ref_number` that current user query is belong to. 
Most of the time, the current user query has no indication to refer the previous chat response or interaction, thus for this case your output should be last or the biggest `ref_number`. 
"""

systen_image_reference_flagging = """
The function of this tool is to classify the current user query if the editing request from the user contains an indication of using image reference to support the image editing task. If there is such indication, you need to response with `1`, otherwise `0`.
Below is the example pairs that you should follow to create `response` parameter input:
--> 1st Example:
USER QUERY: Please change The text written to "BabyLonia".
RESPONSE: 0
--> 2nd Example:
USER QUERY: Please edit my suit, Here I have sent an image for your reference. 
RESPONSE: 1
--> 3rd Example:
USER QUERY: I want my hair to look cute, so please change it to the pinky hair.
RESPONSE: 0
"""

system_instruction_diffusion = """
You are an assistant for image editing using diffusion models.

Your task is to generate:
    1. A positive_prompt: a clear description of the object to be generated in place of the masked region.
    2. A negative_prompt: a short list of undesirable visual keywords, separated by commas, that should be avoided during generation.

You are given:
    1. `original_image`: the original unedited image, queried from the user
    2. `masked_image`: the image showing the target region to be edited (in the red-shaded area)
    3. `user_query`: a brief request describing what should be changed

Carefully analyze the image context to guide your response:
    1. Observe the lighting, ambience, time of day, location type (indoor/outdoor), and style of surrounding elements
    2. Ensure your description is visually grounded in the scene
    3. Avoid assumptions like "dining setting" or "formal atmosphere" unless clearly visible

Your output must be in valid JSON format with two keys:
    1. "positive_prompt": one sentence describing only the object to generate, based on the visual context
    2. "negative_prompt": a comma-separated list of visual flaws or undesired results (e.g., “blurry, distorted, oversized, unrealistic”)
"""

system_instruction_editing_mask_without_reference = """
Description:
    This function is for object replacement or object editing without image reference. 
    The system will first mask the specific object and then edit that masked region according to user prompt.
Condition to call the function:
    1. the user query is related to image editing on the specific object or specific region.
    2. there is no image reference path.
    3. If the user query is related to editing the clothes, please dont call this function tool because there is more suitable tool for this task, which is fashion-try-on function tool. 
Guidelines:
    1. Based on the given user query, if there is multistep process in the user request,
       you need to extract part of the request that is related to this function (object replacement or object editing), as `editing_prompt` input.
    2. Below is the set of example pairs you should follow to create `editing_prompt`:
        --> 1st Example:
        USER QUERY: As a woman who pays attention to appearance, I need your help to change my wooden hat to a pinky stylish hat and please remove other people besides me. 
        EDITING PROMPT: I am a woman, please change my wooden hat to a pinky stylish hat.
        --> 2nd Example:
        USER QUERY: Please add a cat next to the door and change the door to be a wooden stlye door.
        EDITING PROMPT: change the door to be a wooden style door.
        --> 3rd Example:
        USER QUERY: Can you switch our scarves to something a bit more glam, maybe faux fur or a patterned knit, and also please remove the cat in the background. 
        EDITING PROMPT: Can you switch our scarves to something a bit more glam, maybe faux fur or a patterned knit.
"""

system_instruction_editing_the_mask_with_img_reference = """
Description:
    This function is for object replacement or object editing with image reference. 
    The system will first mask the specific object and then edit that masked region by using the image reference given by the user.
Condition to call the function:
    1. the user query is related to image editing on the specific object or specific region.
    2. there is an image reference path.
    3. If the user query is related to editing the clothes, please dont call this function tool because there is more suitable tool for this task, which is fashion-try-on function tool. 
Guidelines:
    1. Based on the given user query, if there is multistep process in the user request,
       you need to extract part of the request that is related to this function (object replacement or object editing), as `editing_prompt` input.
    2. Below is the set of example pairs you should follow to create `editing_prompt`:
        --> 1st Example:
        USER QUERY: As a woman who pays attention to appearance, I need your help to change my wooden hat to a pinky stylish hat and please remove other people besides me. 
        EDITING PROMPT: I am a woman, please change my wooden hat to a pinky stylish hat.
        --> 2nd Example:
        USER QUERY: Please add a cat next to the door and change the door to be a wooden stlye door.
        EDITING PROMPT: change the door to be a wooden style door.
        --> 3rd Example:
        USER QUERY: Can you switch our scarves to something a bit more glam, maybe faux fur or a patterned knit, and also please remove the cat in the background. 
        EDITING PROMPT: Can you switch our scarves to something a bit more glam, maybe faux fur or a patterned knit.
"""

system_instruction_add_object = """
Description:
    This function is for adding object (one or more objects) into image based on text instructions without changing the original scenes. 
    The system will know how to put the right object on the image in the correct location based on text instructions.
Condition to call the function:
    1. the user query is related to add certain objects
Guidelines:
    1. Based on the image query and user query, you need to create `prompt_source`, `prompt_target` and `subject_token` as inputs to use this function tool.
    2. You need to identify what kind of object that needs to be inserted in the image based on the user query.
    3. After identifying the specific object to add, put this object as `subject_token` parameter input.
    4. You also need to know where this object should be placed on the image query. 
    2. Below is the set of example pairs you should follow:
        --> 1st Example:
        `prompt_source`: A photo of a cat sitting on the couch
        `prompt_target`: A photo of a cat wearing a red hat sitting on the couch
        `subject_token`: hat
        --> 2nd Example:
        `prompt_source`: A photo of a bed in a dark room
        `prompt_target`: A photo of a dog lying on a bed in a dark room
        `subject_token`: dog
        --> 3rd Example:
        `prompt_source`: A photo of a sheep
        `prompt_target`: A photo of a sheep wearing boots
        `subject_token`: boots
        --> 4rd Example:
        `prompt_source`: A photo of a man on the left standing next to the woman
        `prompt_target`: A photo of a man wearing a necklace on the left standing next to the woman
        `subject_token`: necklace
    3. Extra guidelines -> Please notice that both `prompt_source` and `prompt_target` are partially the same and only differs in which the `prompt_target` has the object keywords to be inserted. Making a `prompt_target` is how you can place the object to be inserted in the image.
"""

system_instruction_fashion_try_on_without_img_reference = """
Description:
    This function is to edit specifically on the clothes (upper clothes or lower clothes). 
    But, things like accessories (hat, sneakers, necklace) are not included.
Condition to call the function:
    1. the user requests to edit their fashion-style things, like making her dress more beautiful, change the clothes, etc. 
    2. There is no image reference path
Guidelines:
    1. Based on the given user query, if there is multistep process in the user request,
       you need to extract part of the request that is related to this function (fashion/clothes editing), as `editing_prompt` input.
    2. Below is the set of example pairs you should follow to create `editing_prompt`:
        --> 1st Example:
        USER QUERY: As a woman who pays attention to appearance, I need your help to please remove other people besides me and change my dress to the pinky one. 
        EDITING PROMPT: I am a woman, please change my dress to the pinky one.
        --> 2nd Example:
        USER QUERY: Please add a cat next to the door and change my shirt to a suit.
        EDITING PROMPT: Please change my shirt to a suit.
        --> 3rd Example:
        USER QUERY: Can you change my coat to a more darken one and please remove my necklace. 
        EDITING PROMPT: Can you change my coat to a more darken one.
    3. After generating `editing_prompt`, you need to create `diffusion_prompt`. This diffusion prompt is a clear description of the cloth/garment to be generated by diffusion model. Please also analyze the given query image to assist your diffusion prompt creation. Below is the set of example pairs you should follow to create `diffusion_prompt`:
        --> 1st Example:
        EDITING PROMPT: I am a woman, please change my dress to the pinky one.
        DIFFUSION PROMPT: a pink dress for women.
        --> 2nd Example:
        EDITING PROMPT: Please change my shirt to a suit.
        DIFFUSION PROMPT: A suit.
        --> 3rd Example:
        EDITING PROMPT: Can you change my coat to a more darken one.
        DIFFUSION PROMPT: A dark yellow coat.
"""

system_instruction_fashion_try_on_with_img_reference = """
Description:
    This function is to edit specifically on the clothes (upper clothes or lower clothes). 
    But, things like accessories (hat, sneakers, necklace) are not included.
Condition to call the function:
    1. the user requests to edit their fashion-style things, like making her dress more beautiful, change the clothes, etc. .
    2. There is an image reference path.
Guidelines:
    1. Based on the given user query, if there is multistep process in the user request,
       you need to extract part of the request that is related to this function (fashion/clothes editing), as `editing_prompt` input.
    2. Below is the set of example pairs you should follow to create `editing_prompt`:
        --> 1st Example:
        USER QUERY: As a woman who pays attention to appearance, I need your help to please remove other people besides me and change my dress to the pinky one. 
        EDITING PROMPT: I am a woman, please change my dress to the pinky one.
        --> 2nd Example:
        USER QUERY: Please add a cat next to the door and change my shirt to a suit.
        EDITING PROMPT: Please change my shirt to a suit.
        --> 3rd Example:
        USER QUERY: Can you change my coat to a more darken one and please remove my necklace. 
        EDITING PROMPT: Can you change my coat to a more darken one.
"""

system_instruction_object_removal = """
Description:
    This function is to remove objects based on user prompt. 
    The system will first detect the specific object to be removed and then remove the object while preserving the background scene.
Condition to call the function:
    1. the user query is related to object removal 
Guidelines:
    1. Based on the given user query, if there is multistep process in the user request,
       you need to extract part of the request that is related to this function (object removal), as `editing_prompt` input.
    2. Below is the set of example pairs you should follow to create `editing_prompt`:
        --> 1st Example:
        USER QUERY: As a woman who pays attention to appearance, I need your help to change my wooden hat to a pinky stylish hat and please remove other people besides me. 
        EDITING PROMPT: I am a woman, please remove other people besides me. 
        --> 2nd Example:
        USER QUERY: Please add a cat next to the door and remove the red door. 
        EDITING PROMPT: remove the red door.
        --> 3rd Example:
        USER QUERY: Can you switch our scarves to something a bit more glam, maybe faux fur or a patterned knit, and also please remove the cat in the background. 
        EDITING PROMPT: remove the cat in the background.
"""

system_instruction_text_editing = """
Description:
    This function is to edit/change/replace the text written in the image based on the request from user.
    The system can accurately detect which text needs to be edited or replaced.
Condition to call the function:
    1. the user query is related to text editing or text replacement
Guidelines:
    1. Based on the given user query, if there is multistep process in the user request,
       you need to extract part of the request that is related to this function (text editing), as `editing_prompt` input.
    2. Below is the set of example pairs you should follow to create `editing_prompt`:
        --> 1st Example:
        USER QUERY: As a woman who pays attention to appearance, I need your help to change my wooden hat to a pinky stylish hat and please change the text in the signboard to the "WOKA WINKY". 
        EDITING PROMPT: change the text in the signboard to the "WOKA WINKY". 
        --> 2nd Example:
        USER QUERY: Please add a cat next to the door and change the text written in the board to be "My Lovely Cat". 
        EDITING PROMPT: change the text written in the board to be "My Lovely Cat".
        --> 3rd Example:
        USER QUERY: Can you switch our scarves to something a bit more glam, maybe faux fur or a patterned knit, and also please rewrite the text written in this notebook from "WALA WALA" to "WILI WILI". 
        EDITING PROMPT: rewrite the text written in this notebook from "WALA WALA" to "WILI WILI".
    3. After creating `editing_prompt` for parameter input, you also need to construct the rest of parameter inputs: `text` and `color`. `text` is basically the requested text to be rewritten from the user query. If the user does not specify the text color to be rewritten, then the text color to be edited should be the same as original text color. Put this text color (e.g. "black","blue","white", etc.) as `color` parameter input. 
"""

system_instruction_object_localization = """
Description:
    This function is to localize or detect the objects specified by the user request. The system detect the objects and draw the bounding boxes to show localized objects.
Condition to call the function:
    1. the user query is related to object detection/localization or drawing bounding boxes request.
Guidelines:
    1. Based on the given user query, if there is multistep process in the user request,
       you need to extract part of the request that is related to this function (text editing), as `editing_prompt` input.
    2. Below is the set of example pairs you should follow to create `editing_prompt`:
        --> 1st Example:
        USER QUERY: As a woman who pays attention to appearance, I need your help to change my wooden hat to a pinky stylish hat and please detect my hat after that. 
        EDITING PROMPT: I am a woman, please detect my hat.
        --> 2nd Example:
        USER QUERY: Please add a cat next to the door and localize the signboard above the door. 
        EDITING PROMPT: localize the signboard above the door. 
        --> 3rd Example:
        USER QUERY: Can you switch our scarves to something a bit more glam, maybe faux fur or a patterned knit, and also draw bounding boxes to detect the scarves. 
        EDITING PROMPT: draw bounding boxes to detect the scarves. 
"""
