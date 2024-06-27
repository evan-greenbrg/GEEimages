import argparse
import glob
import time
import re
import os
from contextlib import closing
import multiprocessing
from multiprocessing import cpu_count
import ee
import rasterio
from rasterio.merge import merge

# from download import ee_export_image
from helpers import get_download_polygons
from helpers import find_epsg
from ee_datasets import getLandsatCollection
from ee_datasets import getSentinelCollection
from ee_datasets import maskL8sr
from ee_datasets import maskSentinel
from download import ee_export_image

# ee.Authenticate()
ee.Initialize()

NUMBER_OF_PROCESSES = cpu_count() - 1

RES = {
    'landsat': 30,
    'sentinel': 30,
}


def multiprocess(tasks):
    POOL_SIZE = cpu_count() - 1
#    POOL_SIZE = 10

    task_args = []
    for task in tasks:
        fun, args = task
        task_args.append(args)

    results = []
    with closing(multiprocessing.Pool(POOL_SIZE)) as pool:
        completed = pool.starmap(fun, task_args)

    results.extend(completed)

    return results


def pull_image_id(poly, id_string, id_root, poly_i, dataset, dst_crs):
    # See if pausing helpds with the time outs
    time.sleep(6)

    # Get the image
    if dataset == 'landsat':
        allLandsat = getLandsatCollection()
        image = allLandsat.map(
            maskL8sr
        ).filter(ee.Filter.eq(
            'LANDSAT_PRODUCT_ID', id_string
        )).median().clip(
            poly
        )
        image = rescale(image)

    elif dataset == 'sentinel':
        sentinel2 = getSentinelCollection()
        image = sentinel2.map(
            maskSentinel
        ).filter(ee.Filter.eq(
            'DATASTRIP_ID', id_string
        )).median().clip(
            poly
        )

    # Initialize output path
    out_path = os.path.join(
        id_root,
        '{}.tif'
    )
    if not image.bandNames().getInfo():
        return None

    # Get image resolution
    sat_text, date_text = parse_id(id_string, dataset)
    reso = RES[dataset]

    out = out_path.format(
        f'{sat_text}_{date_text}_image_chunk_{poly_i}'
    )

    _ = ee_export_image(
        image,
        filename=out,
        scale=reso,
        crs=dst_crs[0] + ':' + dst_crs[1],
        file_per_band=False
    )

    return out


def mosaic_images(id_root, out_root, sat_text, date_text):
    fps = glob.glob(os.path.join(
        id_root,
        '*image_chunk_*.tif'
    ))
    if not fps:
        return None

    mosaics = []
    for fp in fps:
        ds = rasterio.open(fp)
        mosaics.append(ds)
    meta = ds.meta.copy()
    mosaic, out_trans = merge(mosaics)
    ds.close()

    # Update the metadata
    meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    })

    out_path = os.path.join(
        out_root,
        f'{sat_text}_{date_text}.tif'
    )

    with rasterio.open(out_path, "w", **meta) as dest:
        dest.write(mosaic)

    for fp in fps:
        os.remove(fp)

    return out_path


def parse_id(id_string, dataset):
    if dataset == 'landsat':
        # Formatted sattelite text
        exp = '^(\S{4}_\S{4})'
        m = re.search(exp, id_string)
        sat_text = m.group(1)

        # Formatted Date text
        exp = '_(\d{8})_'
        m = re.search(exp, id_string)
        date_text = m.group(1)

    elif dataset == 'sentinel':
        exp = '^(\S{3})_'
        m = re.search(exp, id_string)
        sat_text = m.group(1)

        # Formatted Date text
        exp = '_(\d{8})T'
        m = re.search(exp, id_string)
        date_text = m.group(1)

    return date_text, sat_text


def rescale(image):
    bns = ['uBlue', 'Blue', 'Green', 'Red', 'Nir', 'Swir1', 'Swir2', 'BQA']
    return image.select(bns).multiply(0.0000275).add(-0.2)


def main(root, polygon_path, dataset, id_file):

    polys = get_download_polygons(polygon_path, root, dataset)

    with open(id_file, 'r') as f:
        id_list = f.readlines()
    id_list = [i.strip() for i in id_list]

    # Get EPSG
    lon, lat = polys[0].getInfo()['coordinates'][0][0]
    dst_crs = find_epsg(lat, lon)

    # Download the chunks
    tasks = []
    # Iterate over all the IDS to download
    for id_i, id_string in enumerate(id_list):
        print(f'Pulling Image ID: {id_string}')
        for poly_i, poly in enumerate(polys):
            id_root = os.path.join(
                root, id_string,
            )
            os.makedirs(id_root, exist_ok=True)

            tasks.append((
                pull_image_id,

                (
                    poly, id_string, id_root, poly_i, dataset, dst_crs
                )
            ))
    multiprocess(tasks)

    # Mosaic all the images
    for id_i, id_string in enumerate(id_list):
        id_root = os.path.join(
            root, id_string,
        )

        sat_text, date_text = parse_id(id_string, dataset)

        out_root = os.path.join(
            root, f'{sat_text}_{date_text}',
        )
        os.makedirs(out_root, exist_ok=True)

        out_fp = mosaic_images(
            id_root, out_root, sat_text, date_text
        )

        if not out_fp:
            continue

        os.rmdir(id_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pull Images from List')

    parser.add_argument('--root', metavar='root', type=str,
                        help='Base root for image pulling')

    parser.add_argument('--poly', metavar='poly', type=str,
                        help='Path to the polygon to pull')

    parser.add_argument('--dataset', metavar='dataset', type=str,
                        help='Which GEE dataset to use')

    parser.add_argument('--id_file', metavar='id_file', type=str,
                        help='Path to the txt file with the ids to pull')

    args = parser.parse_args()
    main(args.root, args.poly, args.dataset, args.id_file)

# root = '/home/greenberg/ExtraSpace/Misc/Port'
# polygon_path = os.path.join(root, 'Baltimore.gpkg')
# dataset = 'sentinel'
# polys = get_download_polygons(polygon_path, root, dataset)
# # id_string = 'S2A_OPER_MSI_L2A_DS_VGS1_20220311T202457_S20220311T160207_N04.00'
# # dataset = 'landsat'
# # polys = get_download_polygons(polygon_path, root, dataset)
# id_file = '/home/greenberg/Code/Github/GEEimages/file_list.txt'
