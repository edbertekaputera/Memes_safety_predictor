from PIL import Image
import pytesseract
import preprocess_image as ppImg
import translate_image as trnsImg

def extractText(img_path):
    #extract image
    image = Image.open(img_path)

    #create preprocess instances
    preprocessorBasic = ppImg.PreprocessImage(metrics=['grayscale','bilateral','thresholding'])
    preprocessorEng = ppImg.PreprocessImage(metrics=['grayscale','remove_noise','thresholding'])
    preprocessorChi = ppImg.PreprocessImage(metrics=['grayscale','remove_noise'])
    preprocessorTan = ppImg.PreprocessImage(metrics=['grayscale','thresholding'])

    #detect language
    image_np = preprocessorBasic.transform_image(image)
    converted_image = Image.fromarray(image_np)
    script_name, _ = trnsImg.detect_language(img_path)

    #select language
    if script_name == "Han":
        image_npChi = preprocessorChi.transform_image(image)
        converted_imageChi = Image.fromarray(image_npChi)

        text = pytesseract.image_to_string(converted_imageChi, lang='chi_sim')
        lang = "chi_sim"
    elif script_name == "Tamil":
        image_npTam = preprocessorTan.transform_image(image)
        converted_imageTam = Image.fromarray(image_npTam)

        text = pytesseract.image_to_string(converted_imageTam, lang='tam')
        lang = "Tam"
    elif script_name == "Arabic":
        image_npEng = preprocessorEng.transform_image(image)
        image_npChi = preprocessorChi.transform_image(image)
        image_npTam = preprocessorTan.transform_image(image)

        converted_imageEng = Image.fromarray(image_npEng)
        converted_imageChi = Image.fromarray(image_npChi)
        converted_imageTam = Image.fromarray(image_npTam)

        text1 = pytesseract.image_to_string(converted_imageChi, lang='chi_sim')
        text2 = pytesseract.image_to_string(converted_imageTam, lang='tam')
        text3 = pytesseract.image_to_string(converted_image, lang='eng')
        if len(text1) < len(text2)/1.5:
            if len(text3) < 3*len(text2):
                text = text2
                lang = "Tam"
            else:
                text = text3
                lang = "eng"
        else:
            if len(text3) < len(text1):
                text = text1
                lang = "chi_sim"
            else:
                text = text3
                lang = "eng"
    else:
        image_npEng = preprocessorEng.transform_image(image)
        converted_imageEng = Image.fromarray(image_npEng)

        text = pytesseract.image_to_string(converted_imageEng, lang='eng')
        lang = "eng"
    
    return text, lang