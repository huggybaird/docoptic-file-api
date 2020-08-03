import uvicorn
from fastapi import FastAPI, File, UploadFile, Body
from pydantic import BaseModel, Field# pylint: disable=no-name-in-module
from PIL import Image, ImageSequence
import io
from starlette.responses import StreamingResponse
from fastapi.responses import RedirectResponse, FileResponse, StreamingResponse
# from fastapi.responses import FileResponse
# from tempfile import NamedTemporaryFile
# from shutil import copyfileobj
import json
from pdf2image import convert_from_path, convert_from_bytes
import numpy as np
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
from typing import List, Optional
import pytesseract
import cv2
from fastapi.middleware.cors import CORSMiddleware
import pdfkit
import tempfile

# nltk.set_proxy('http://proxy.example.com:3128', ('USERNAME', 'PASSWORD'))
# nltk.download('stopwords', download_dir=r"D:\Dropbox\dev\docoptic-file-api\resources")
nltk.data.path.append(r"D:\Dropbox\dev\docoptic-file-api\resources")
stopwords = stopwords.words('english')

# CORS (Cross-Origin Resource Sharing) 
origins = [
    "http://myocpapp.pncint.net", 
    "https://myocpapp.pncint.net", 
    "http://localhost",
    "http://localhost:4200",
    "http://localhost:8080",
]

app = FastAPI(
    title="DocOptic File Processing API",
    description="""Open source API's for file processing including thumbnails, OCR, text extraction, and compression
        \n To see this documentation in Redoc.ly format go to [/redoc](./redoc)
        """,
    version="0.2.0",
    docs_url="/swagger-ui.html", 
    redoc_url="/redoc"

)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.post("/files/")
# def create_file(file: bytes = File(...)):
#     return {"file_size": len(file)}

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile = File(...)):
#     a = 1
#     return {"filename": file.filename, "fileSize": str(file.spool_max_size/1024)+" KB"}

# @app.get("/domath/")
# def get_do_math():
#     v = np.array([3, 7])
#     u = np.array([2, 2])
#     myResult = v + u   
#     print(myResult)
#     return {"math-result": str(myResult)}



@app.get("/", include_in_schema=False)
def read_root():
    # return {"Hello": "World"}
    return RedirectResponse("./swagger-ui.html")

@app.get("/testConnectivity/", tags=["default"]
    ,summary="Test connectivity with this simple 'GET'"
    ,description="""Use this method as a ***sanity check*** to ensure you get a "SUCCESS" response.""")
def test_connectivity():
    # return {"Hello": "World"}
    return {"status": "SUCCESS", "detailedMessage": "You have successfully connected to the docoptic file processing API!  Contradulations!"}


@app.post("/document/image/extractText/",  tags=["document"], #response_model=List[DocumentSimilarityCalculation], 
summary="Use tesseract OCR to extract text from an image",
    description="""Returns a string of text extracted from the image using [pytesseract](https://pypi.org/project/pytesseract/)
    \n Supported file (aka mime) types:
    \n * image/***tiff***   
    \n * application/***pdf***   
    \n * image/***bmp***  
    \n * image/***gif***  
    \n * image/***jpeg*** 
    \n * image/***png***  
    \n """
    ,response_description="The extracted text returned as a string"
    , response_model_exclude_unset=True
    )
def document_image_extract_text(file: UploadFile = File(...)):
    img=Image.open(file.file) #"images/ocr.jpg")
    # img=Image.open(r"D:\Dropbox\pnc\api-enablement\api-first-video\graphics\tiff and pdf\simple-letters.png")
    
    # # Grayscale image
    # img = Image.open(file.file).convert('L')
    # ret,img = cv2.threshold(np.array(img), 125, 255, cv2.THRESH_BINARY)

    # # Older versions of pytesseract need a pillow image
    # # Convert back if needed
    # img = Image.fromarray(img.astype(np.uint8))


    # image = cv2.imread(file.file, cv2.IMREAD_GRAYSCALE) 
    # cv2.dilate(image, (5, 5), image)

    # print("*****************************************************************************")
    # print(pytesseract.image_to_string(image), config='--psm 7') 

    result = pytesseract.image_to_string(img)

    print("*****************************************************************************")
    print("CONTENT TYPE", file.content_type)
    fileFormat = mimeTypeDict.get(file.content_type).get("thumbnail-format").upper()
    print("FILE FORMAT", fileFormat)
    print("*****************************************************************************")
    print(result)

    return {result}

    # return "result: {}".format(result)



class DocumentCompareSecondary(BaseModel):
    documentId: str 
    documentExtractedText: str  

class DocumentComparePrimary(BaseModel):
    primaryDocumentId: str = Field(None, title="The unique identifier for the primary document", max_length=300)
    primaryDocumentExtractedText: str 
    compareToDocuments: List[DocumentCompareSecondary] = []
    class Config:
            orm_mode = True
            schema_extra = {
                'example': {
                        'primaryDocumentId': "primary.pdf",
                        'primaryDocumentExtractedText': """This is the primary document I care about.  This may be a revision so lets compare how similar the text is to other files?  Let's find out by comparing similar words"""
                        ,'compareToDocuments': [
                            {
                                'documentId': 'doc_to_compare1.doc',
                                'documentExtractedText':  """I have NOTHING in common except the word primary"""
                            },
                            {
                                'documentId': 'doc_to_compare2.doc',
                                'documentExtractedText': "The is SIMILAR to the primary document I care about. This may be a revision so notice how this compares very similar to the text in the primary documet. By comparing similar words, the percentageSimilar will be higher"
                            },
                            {
                                'documentId': 'doc_to_compare3.doc',
                                'documentExtractedText': """I AM THE SAME EXCEPT FIRST SENTENCE. This is the primary document I care about.  This may be a revision so lets compare how similar the text is to other files?  Let's find out by comparing similar words"""
                            }
                        ]
                    } 
            }

class DocumentSimilarityCalculation(BaseModel):
    documentId: str   
    percentageSimilar: float 
    class Config:
            orm_mode = True
            schema_extra = {
                'example': [
                                {
                                    "documentId": "doc_to_compare1.doc",
                                    "percentageSimilar": 0.1025978352085154
                                },
                                {
                                    "documentId": "doc_to_compare2.doc",
                                    "percentageSimilar": 0.6882472016116852
                                },
                                {
                                    "documentId": "doc_to_compare3.doc",
                                    "percentageSimilar": 0.9293203772845849
                                }
                            ]
            } 

# response_model=DocumentSimilarityCalculatoin,
@app.post("/document/calculateSimilarity/",  tags=["document"], response_model=List[DocumentSimilarityCalculation], 
summary="Calculate similarity of words between documents",
    description="""Returns a percentage (%) of how closely the text match between documents.
    \n Often documents have revisions and 95% of the text is similar; 
    \n You pass in the ***Primary Document's*** extracted text.  You also pass in the extracted text from multiple other documents to compare to.  The response will give a percentageSimilar of how each of the ***compareToDocuments*** words match the primary documents words"""
    ,response_description="A list of how similar each dodument is to the primary document"
    , 
    )
def post_calculate_similarity( 
    docCompareList: DocumentComparePrimary
#     string1: str = """Now I’ll declare a list of some arbitrary, and hard to find,  to a degree similar sentences. It’s a list because the vector space will be created from all unique words, and this will ensure that every vector has the same number of dimensions — as you cannot calculate the angle between vectors in different dimensional space"""
#     ,string2: str = """Remember the imports? We imported quite a lot, and it’s time to make use of it. I’ll declare a function which will do the following:
# Remove punctuations from a given string
# Lowercase the string
# Remove stopwords
# Now I’ll declare a list of some arbitrary, and hard to find,  to a degree similar sentences. It’s a list because the vector space will be created from all unique words, and this will ensure that every vector has the same number of dimensions — as you cannot calculate the angle between vectors in different dimensional space"""
#     ,string3: str = "Great, the preparation is almost done. Now you’ll utilize the power of CountVectorizer to perform some magic (not really). It will create k vectors in n-dimensional space, where k is the number of sentences, and n is the number of unique words in all sentences combined. Then, if a sentence contains a certain word, the value will be 1 and 0 otherwise"
    ):
    # sentences = [string1, string2, string3]
    # primaryText = docCompareList.primaryDocumentExtractedText
    response = []
    for i, document in enumerate(docCompareList.compareToDocuments): 
        print("document ",i," id",docCompareList.compareToDocuments[i].documentId )
        sentences = [docCompareList.primaryDocumentExtractedText, document.documentExtractedText]  
        # return {"math-result": str(a)}
        cleaned = list(map(clean_string, sentences))
        # print(cleaned)
        # print("************************************************************************************")
        vectorizer = CountVectorizer().fit_transform(cleaned)
        vectors = vectorizer.toarray()
        # print(vectors)
        # print("************************************************************************************")
        csim = cosine_similarity(vectors)
        resultSimilarity = cosine_sim_vectors(vectors[0], vectors[1])
        response.append({"documentId":document.documentId, "percentageSimilar":resultSimilarity})
    print(json.dumps(response))
    # return [{"documentId":1234, "percentageSimilar":99.99}]
    return response #json.dumps(response) #{"math-result": str(resultSimilarity)}

def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text=' '.join([word for word in text.split() if word not in stopwords])
    return text

 
@app.post("/document/image/resize/",  tags=["document"], #response_model=List[DocumentSimilarityCalculation], 
summary="Resize and image (i.e. generate a thumbnail)",
    description="""Returns a resized image based on the ***height*** and ***width*** parameters passed in.
    \n Supported file (aka mime) types:
    \n * image/***tiff*** - returns a ***jpeg*** thumbnail to reduce size 
    \n   * supports passing in a ***page number*** to generate the thumbnail from 
    \n * application/***pdf*** - returns a ***jpeg*** thumbnail to reduce size 
    \n   * supports passing in a ***page number*** to generate the thumbnail from 
    \n * image/***bmp***  - returns a ***jpeg*** thumbnail to reduce size 
    \n * image/***gif*** - returns a ***gif*** (not animated) thumbnail
    \n * image/***jpeg*** - returns a ***jpeg*** thumbnail
    \n * image/***png***  - returns a ***png*** thumbnail
    \n This is useful to "preview" pages from a document/image in a small picture/tile/thumbnail form
    \n ***FUTURE TODO*** Currently the "endPage is ignored.  In other words, this API ony extracts one page based on the ***startPage***. In the future, we will add a feature to extract all pages between the ***startPage*** and ***endPage*** and return them as a List[]"""
    ,response_description="A list of how similar each dodument is to the primary document"
    , response_model_exclude_unset=True
    )
# @app.post("/image/resize/")
async def create_image_resize(file: UploadFile = File(...),
    thumbnailHeight: int = 660, thumbnailWidth: int = 420, startPage: int = 1, endPage: int=10):

    # convert pdf to images
    if file.content_type.lower() == "application/pdf":
        im = convert_from_bytes(file.file.read()) 
        #  images = convert_from_bytes(open('/home/belval/example.pdf', 'rb').read())
        # im = Image.open(pdfImage)
    else:
        im = Image.open(file.file)

    # im = Image.open(file.file)
    # im = resize_image(im,thumbnailHeight, thumbnailWidth)


    print("CONTENT TYPE", file.content_type)
    fileFormat = mimeTypeDict.get(file.content_type).get("thumbnail-format").upper()
    print("FILE FORMAT", fileFormat)
    print("*****************************************************************************")

    # b = io.BytesIO()
    b = []
    if file.content_type.lower() == "application/pdf":
        for i, page in enumerate(im): 
            # page.save("%s-page%d.jpg" % (pdf_file,pages.index(page)), "JPEG")
            if i >= startPage-1 and i < endPage: #== 4:
                page = resize_image(page,thumbnailHeight, thumbnailWidth)
                b.append(io.BytesIO()) 
                page.save(b[-1], format=fileFormat) #page.save(b, format=fileFormat) #.save("page%d.png" % i)
    else:
        for i, page in enumerate(ImageSequence.Iterator(im)):
            if i >= startPage-1 and i < endPage: #== 4:
                page = resize_image(page,thumbnailHeight, thumbnailWidth)
                b.append(io.BytesIO()) 
                page.save(b[-1], format=fileFormat) #page.save(b, format=fileFormat) #.save("page%d.png" % i)

    # im.save(b, format=fileFormat) #format="JPEG") #format="PNG") #format="JPEG") #"JPEG", quality=80) #"JPEG")
    # b.name = "hello.jpg" 
    # return StreamingResponse(b, media_type="image/jpeg",
    # headers={'Content-Disposition': 'inline; filename="hello.jpg"'})
    b[0].seek(0) #Important for streams to work 
    return StreamingResponse(b[0], media_type= mimeTypeDict.get(file.content_type).get("return-mime-type")) #media_type=file.content_type)  


def resize_image(image: Image, thumbnailHeight: int, thumbnailWidth: int) :
    SIZE = (thumbnailHeight, thumbnailWidth) # (220, 140) 
    image.thumbnail(SIZE, Image.ANTIALIAS)
    return image

# def resize_image(file: UploadFile, thumbnailHeight: int, thumbnailWidth: int) :
#     SIZE = (thumbnailHeight, thumbnailWidth) # (220, 140) 
#     im = Image.open(file.file)
#     # if im.mode in ("RGBA", "P"): 
#     #     im = im.convert("RGB")
#     # im = im.convert('RGB')
#     im.thumbnail(SIZE, Image.ANTIALIAS)
#     return im
 
    # tempFile =  NamedTemporaryFile(mode='w+b',suffix='jpg')
    # im.save(tempFile, 'JPEG', quality=80)
    # file_like = open(tempFile, mode="rb")
    # return FileResponse(some_file_path)
    # return StreamingResponse(file_like, media_type=file.content_type) #"video/mp4")


    # im.save(r"D:\Dropbox\dev\docoptic-file-api\myjpg.jpg", 'JPEG', quality=80)
    # file_like =  NamedTemporaryFile(mode='w+b',suffix='jpg')
    # im.save(file_like, 'JPEG', quality=80)
    # return StreamingResponse(iter(im), media_type=file.content_type) #"video/mp4")


    # imgByteArr = io.BytesIO()
    # tempFileObj = NamedTemporaryFile(mode='w+b',suffix='jpg')
    # im.save(tempFileObj, 'JPEG') # format=file.content_type)
    # # copyfileobj(im,tempFileObj)
    # # imgByteArr = imgByteArr.getvalue()

    # # file_like = open(imgByteArr, mode="rb")
    
    # return StreamingResponse(tempFileObj, media_type=file.content_type) #"video/mp4")

    # b = io.BytesIO(b"")
    
    # return StreamingResponse(file_like, media_type=file.content_type) #"video/mp4")

    # return FileResponse(file.file)
    # return StreamingResponse(file.file.readable, media_type=file.content_type) #"video/mp4")
    
    # return {"filename": file.filename, "fileSize": str(file.spool_max_size/1024)+" KB"}

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None




# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: str = None):
#     return {"item_id": item_id, "q": q}


# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_price": item.price, "item_id": item_id}


mimeTypeDict =  {
  "image/bmp": {"thumbnail-format": "jpeg", "return-mime-type":"image/jpeg"}, #"bmp",
  "image/gif": {"thumbnail-format": "gif", "return-mime-type":"image/gif"},
  "image/x-icon": {"thumbnail-format": "ico", "return-mime-type":"image/x-icon"},
  "image/jpeg":  {"thumbnail-format": "jpeg", "return-mime-type":"image/jpeg"},
  "image/png": {"thumbnail-format": "png", "return-mime-type":"image/png"},
  #"image/svg+xml": {"thumbnail-format": "svg", "return-mime-type":"image/svg+xml"},
  "image/tiff":  {"thumbnail-format": "jpeg", "return-mime-type":"image/jpeg"}, #"tif"
  "application/pdf":  {"thumbnail-format": "jpeg", "return-mime-type":"image/jpeg"} 
}




class HtmlRender(BaseModel):
    htmlTemplate: str
    jsonData: str
    class Config:
            orm_mode = True
            schema_extra = {
                'example': {
                        'htmlTemplate':"""<html>
                        <head></head>
                        <body>
                            <h1>sample {{{mySampleTitle}}} </h1>
                            <p>{{{##a}}}. Here is a sentance with a {{{sampleValue1}}} replaced by the generator</p>
                            {{{IF state IN [OH,FL,CA,NY]}}}
                            <p>{{{##a}}}. Here is a paragraph that prints because {{{state}}} is in list [OH,FL,CA,NY]. </p>
                            {{{END IF state}}}
                            <p>{{{##a}}}. Here is a second paragraph being auto-numbered</p>
                            {{{FOR EACH loans}}} 
                            <p>&nbsp;&nbsp;{{{##loanNumber}}}. Account number {{{loans.loanNumber}}}} has a balance of {{{loans.loanBalance}}} </p>
                            {{{END FOR EACH loans}}}
                        </body>
                        </html>""",
                        'jsonData':  """
                            {
                                "mySampleTitle": "Awesomeness is here!",
                                "sampleValue1": "$100,000.00",
                                "state": "OH",
                                "loans":[
                                    {"loanNumber":"12345", "loanBalance":"$200,000"},
                                    {"loanNumber":"67890", "loanBalance":"$999"}
                                ]
                            }
                            """
                    } 
            }
    
@app.post("/document/generate/html/",  tags=["document generator"],   
summary="Parses the json and fills in the html template",
    description="""Processes the html document with the {{{fields}}} populated from the json object
     Document generation in html"""
    ,response_description="Returns a string. the string representes the generated source html  "
    , 
    )
def post_generate_html( htmlRender: HtmlRender
    # htmlTemplate: str =  """<html>
    #     <head></head>
    #     <body>
    #         <h1>sample {{{mySampleTitle}}} </h1>
    #         <p>{{{##a}}}. Here is a sentance with a {{{sampleValue1}}} replaced by the generator</p>
    #         {{{IF state IN [OH,FL,CA,NY]}}}
    #         <p>{{{##a}}}. Here is a paragraph that prints because {{{state}}} is in list [OH,FL,CA,NY]. </p>
    #         {{{END IF state}}}
    #         <p>{{{##a}}}. Here is a second paragraph being auto-numbered</p>
    #         {{{FOR EACH loans}}} 
    #         <p>&nbsp;&nbsp;{{{##loanNumber}}}. Account number {{{loans.loanNumber}}}} has a balance of {{{loans.loanBalance}}} </p>
    #         {{{END FOR EACH loans}}}
    #     </body>
    #     </html>""",
    # jsonData: str = """
    # {
    #     "mySampleTitle": "Awesomeness is here!",
    #     "sampleValue1": "$100,000.00",
    #     "state": "OH",
    #     "loans":[
    #         {"loanNumber":"12345", "loanBalance":"$200,000"},
    #         {"loanNumber":"67890", "loanBalance":"$999"}
    #     ]
    # }
    # """
    ):
    # sentences = [string1, string2, string3]
    # primaryText = docCompareList.primaryDocumentExtractedText
    htmlTemplate = htmlRender.htmlTemplate
    jsonDataStr = htmlRender.jsonData
    responseHtml: str = htmlTemplate
    print("try parsing json")
    json_object = json.loads(jsonDataStr)
    print("try recursive loop over json")
    for (path, node) in traverse(json_object): 
        path_complete = '.'.join(map(str, path))
        value_complete = ''.join(map(str, node)) 
        # print("Path:" + path_complete + ", Value:"+value_complete + "[" + str(type(dict)) +","+ str(type(node))+"]")
        if type(node) == str:
            ##########################################################################################
            #  Replace {{{attribute}}}
            ##########################################################################################
            responseHtml = responseHtml.replace("{{{" + path_complete + "}}}", value_complete)
            print("ValueNode:" + path_complete)
            
            ##########################################################################################
            #  Replace {{{IF attribute IN [value1, value2, value3]}}} 
            #  {{{END IF}}}
            ########################################################################################## 
            token = get_start_token(responseHtml,"{{{IF "+path_complete + " IN ")
            while(token != ""):
                startIndex = responseHtml.find(token) 
                # print("IF token:",token, " start:", startIndex)
                endIndex = responseHtml.find("{{{END IF " + path_complete + "}}}", startIndex) + len("{{{END IF " + path_complete + "}}}")
                ifStatementHtml = responseHtml[startIndex: endIndex]
                print("ifStatementHtml:",ifStatementHtml)
                ifStatementParsed = ""
                if(token.find(value_complete, token.find("[")) > 0):
                    ifStatementParsed = ifStatementHtml.replace(token,"")
                    ifStatementParsed = ifStatementParsed.replace("{{{END IF " + path_complete + "}}}","")
                responseHtml = responseHtml.replace(ifStatementHtml, ifStatementParsed)
                token = get_start_token(responseHtml,"{{{IF "+path_complete + " IN ")

        else:
            #########################################################################################
            # Replace {{{FOR EACH attribute}}} and {{{END FOR EACH attribute}}}
            # This loops through the list and populates the html
            # ########################################################################################
            print("List or Dictionary:" + path_complete)
            # TODO: still need to program this functionality for the 

    ################################################################################
    #  Replace all paragraph numbers {{{##a}}}, {{{##b}}}
    ################################################################################
    if(responseHtml.find("{{{##")): 
        while(get_start_token(responseHtml,"{{{##") != ""):
            token = get_start_token(responseHtml,"{{{##")
            print("Token:",token)
            tokenNumber = 1 
            while(responseHtml.find(token) > 0 and tokenNumber<999):
                print("replacing tokenNumber:", tokenNumber)
                responseHtml = responseHtml.replace(token, str(tokenNumber),1)
                tokenNumber = tokenNumber + 1 

    return responseHtml 

def traverse(dict_or_list, path=[]):
    if isinstance(dict_or_list, dict):
        iterator = dict_or_list.items()
    else:
        iterator = enumerate(dict_or_list)
    for k, v in iterator:
        yield path + [k], v
        if isinstance(v, (dict, list)):
            for k, v in traverse(v, path + [k]):
                yield k, v

#############################################################
#  get token looks returns {{{myToken}}} from an html string
#  examples: {{{myAttribute}}}, {{{##a}}}, 
def get_start_token(html: str, startOfTokenString: str):
    responseToken = ""
    startOfToken = html.find(startOfTokenString)
    print("startOfToken:",startOfToken)
    endOfToken =  html.find("}}}",startOfToken)+3
    # print("endOfToken:",endOfToken)
    responseToken = html[ startOfToken : endOfToken]
    return responseToken

# def recursive_json_loop(dict):
#     print("looping inside2")
#     for key, value in dict.items(): 
#         if type(value) == type(dict):
#             if key != "paging":
#                 for key, value in value.items():
#                     if isinstance (value,list):
#                         print(key)
#                         # place where I need to enter list comprehension?
#                 if type(value) == type(dict):
#                     if key == "id":
#                         print(" id found " + value)
#                     if key != "id":
#                         print(key + " 1st level")
#                 if key == "id":
#                     print(key)
#         else:
#             if key == "id":
#                 print("id found " + value)

 
# def walk(d):
#     global path
#       for k,v in d.items():
#           if isinstance(v, str) or isinstance(v, int) or isinstance(v, float):
#             path.append(k)
#             print "{}={}".format(".".join(path), v)
#             path.pop()
#           elif v is None:
#             path.append(k)
#             ## do something special
#             path.pop()
#           elif isinstance(v, dict):
#             path.append(k)
#             walk(v)
#             path.pop()
#           else:
#             print "###Type {} not recognized: {}.{}={}".format(type(v), ".".join(path),k, v)

# mydict = {'Other': {'Stuff': {'Here': {'Key': 'Value'}}}, 'root1': {'address': {'country': 'Brazil', 'city': 'Sao', 'x': 'Pinheiros'}, 'surname': 'Fabiano', 'name': 'Silos', 'height': 1.9}, 'root2': {'address': {'country': 'Brazil', 'detail': {'neighbourhood': 'Central'}, 'city': 'Recife'}, 'surname': 'My', 'name': 'Friend', 'height': 1.78}}

# path = []
# walk(mydict)

class PdfGenerator(BaseModel):
    title: Optional[str] = None
    sourceReferenceNumber: Optional[str] = None
    targetPdfFileName: Optional[str] = None
    htmlSource: str
    class Config:
            orm_mode = True
            schema_extra = {
                'example': {
                        'title': "My Cool Document",
                        'sourceReferenceNumber': "Guid-or-Id-for-tracing",
                        'targetPdfFileName': "my-generated-filename.pdf",
                        'htmlSource':"""<p><img alt="pnc bank" src="https://1000logos.net/wp-content/uploads/2019/12/PNC-Bank-Logo.png" style="height:113px; width:200px" /></p> <h1>Final With Image</h1> <p>Here is an example of how to populate the price of <span style="color:#1abc9c"><span style="font-size:36px">$200.99</span></span> for the <span style="color:#c0392b">product</span>.&nbsp;</p> <p>Here is another <strong>Sally Smith</strong></p> <p></p> <p>Here is some verbiage specific to states NY, PA, OH</p> <p></p> <p>1. This is the first paragraph</p> <p>2. This is the second paragraph</p> <p>3. This is the third paragraph</p> <p>Here is an example of a list:</p> <p>{{{FOR EACH loans}}}</p> <p>Account number {{{loans.loanNumber}}}} has a balance of {{{loans.loanBalance}}} and matures on {{{loans.loanMaturity}}}</p> <p>{{{END FOR EACH loans}}}</p>
                            """
                    } 
            }
    
@app.post("/document/generate/pdf/",  tags=["document generator"],   
summary="Converts html to PDF",
    description="""Processes the html document with pdfkit and wk<html>toPDF more info https://wkhtmltopdf.org/"""
    ,response_description="Returns a pdf document "
    , 
    )
def post_generate_pdf(pdfGenerator: PdfGenerator
# html: str =  """<p><img alt="pnc bank" src="https://1000logos.net/wp-content/uploads/2019/12/PNC-Bank-Logo.png" style="height:113px; width:200px" /></p> <h1>Final With Image</h1> <p>Here is an example of how to populate the price of <span style="color:#1abc9c"><span style="font-size:36px">$200.99</span></span> for the <span style="color:#c0392b">product</span>.&nbsp;</p> <p>Here is another <strong>Sally Smith</strong></p> <p></p> <p>Here is some verbiage specific to states NY, PA, OH</p> <p></p> <p>1. This is the first paragraph</p> <p>2. This is the second paragraph</p> <p>3. This is the third paragraph</p> <p>Here is an example of a list:</p> <p>{{{FOR EACH loans}}}</p> <p>Account number {{{loans.loanNumber}}}} has a balance of {{{loans.loanBalance}}} and matures on {{{loans.loanMaturity}}}</p> <p>{{{END FOR EACH loans}}}</p>
# """
):
    print("generate pdf")
    # some_file_path = pdfGenerator.targetPdfFileName # 'generated-doc.pdf'

    temp_name = tempfile.gettempdir() + "/" + next(tempfile._get_candidate_names())
    print(temp_name)
    pdfkit.from_string(pdfGenerator.htmlSource, temp_name) # some_file_path)
    # return FileResponse('micro.pdf')
    # file_like = open(some_file_path, mode="rb")
    response = FileResponse(path=temp_name, filename=pdfGenerator.targetPdfFileName) # some_file_path)
    return response
    if os.path.exists(temp_name):
        os.remove(temp_name)
        print("DELETE:",temp_name)
    # return FileResponse(file_like.name, media_type="application/pdf")
    # return StreamingResponse(file_like, media_type="application/pdf")  

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)