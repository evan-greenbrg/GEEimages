import os
import ee
import ee.mapclient
import numpy as np

# ee.Authenticate()
ee.Initialize()


def maskL8sr(image):
    """
    Masks out clouds within the images
    """
    # Bits 3 and 5 are cloud shadow and cloud
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    # Get pixel QA band
    qa = image.select('BQA')
    # Both flags should be zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
        qa.bitwiseAnd(cloudsBitMask).eq(0)
    )
    return image.updateMask(mask)


def maskSentinel(image):
    cloudShadowBitMask = (1 << 10)
    cloudsBitMask = (1 << 11)

    qa = image.select('BQA')

    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
        qa.bitwiseAnd(cloudsBitMask).eq(0)
    )

    return image.updateMask(mask)


def clip(image, polygon):
    return image.clip(polygon)


def getSentinelCollection():
    bnSen = ['B2', 'B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'QA60',]
    bns = ['uBlue', 'Blue', 'Green', 'Red', 'Nir', 'Swir1', 'Swir2', 'BQA']

    sen = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").select(bnSen, bns)

    return (sen)


def getLandsatCollection():
    """
    merge landsat 5, 7, 8 collection 1
    tier 1 SR imageCollections and standardize band names
    """
    bn9 = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL',]
    bn8 = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL',]
    bn7 = ['SR_B1', 'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL',]
    bn5 = ['SR_B1', 'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL',]
    bns = ['uBlue', 'Blue', 'Green', 'Red', 'Nir', 'Swir1', 'Swir2', 'BQA']

    # create a merged collection from landsat 5, 7, and 8
    ls5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").select(bn5, bns)

    ls7 = (
        ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterDate(
            '1999-04-15', '2003-05-30'
        ).select(bn7, bns)
    )

    #ls8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR").select(bn8, bns)
    ls8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(bn8, bns)

    ls9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").select(bn9, bns)

    #merged = ls5.merge(ls7).merge(ls8).merge(ls9)
    merged = ls5.merge(ls7).merge(ls8).merge(ls9)

    return (merged)


def get_images(polygon, start, end, dataset='landsat'):
    if dataset == 'landsat':
        allLandsat = getLandsatCollection()
        images = allLandsat.map(
            maskL8sr
        ).filterDate(
            start, end 
        # ).filterBounds(polygon)
        ).filterBounds(polygon).map(
            lambda image: clip(image, polygon)
        )
        images = ee.ImageCollection(images)

    elif dataset == 'sentinel':
        sentinel2 = getSentinelCollection()
        images = sentinel2.map(
            maskSentinel 
        ).filterDate(
            start, end
        #).filterBounds(polygon)
        ).filterBounds(polygon).map(
            lambda image: clip(image, polygon)
        )

    return images


def get_image_period(start, end, polygon, dataset='landsat'):
    images = ee.List([])
    if dataset == 'landsat':
        allLandsat = getLandsatCollection()
        images = images.add(allLandsat.map(
            maskL8sr
        ).filterDate(
            start, end 
        ).median().clip(
            polygon
        ))
        images = ee.ImageCollection(images)

    elif dataset == 'sentinel':
        sentinel2 = getSentinelCollection()
        images = images.add(sentinel2.map(
            maskSentinel 
        ).filterDate(
            start, end
        ).median().clip(
            polygon
        ))
        images = ee.ImageCollection(images)

    return images.median().clip(polygon)


def request_params(filename, scale, image):
    filename = os.path.abspath(filename)
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]

    params = {"name": name, "filePerBand": False}
    params["scale"] = scale
    region = image.geometry()
    params["region"] = region

    return params

