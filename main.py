import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field# pylint: disable=no-name-in-module
from PIL import Image, ImageSequence
import io
from starlette.responses import StreamingResponse
from fastapi.responses import RedirectResponse, FileResponse
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
from typing import List
import pytesseract
import cv2

# nltk.set_proxy('http://proxy.example.com:3128', ('USERNAME', 'PASSWORD'))
nltk.download('stopwords')
stopwords = stopwords.words('english')

app = FastAPI(
    title="DocOptic File Processing API",
    description="""Open source API's for file processing including thumbnails, OCR, text extraction, and compression
        \n To see this documentation in Redoc.ly format go to [/redoc](./redoc)
        """,
    version="0.2.0",
    docs_url="/swagger-ui.html", 
    redoc_url="/redoc"

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)