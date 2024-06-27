import argparse
import datetime
import re

from affine import Affine
import ee
import numpy as np
import pandas
from pyproj import Transformer
import rasterio
from shapely import geometry

from helpers import get_polygon
from ee_datasets import get_images

# ee.Authenticate()
ee.Initialize()


def get_coverage(image, full_coverage, poly):
    zeros = image.Not()
    ones = zeros.Not()

    non_zero_px = ones.reduceRegion(
      ee.Reducer.sum(),
      poly,
      30,
    ).getInfo()['Blue']

    dims = image.getInfo()['bands'][0]['dimensions']
    size = dims[0] * dims[1]

    image_coverage = 100 * (non_zero_px / size)
    return image_coverage / full_coverage


def get_full_coverage(shape, epsg, dims):
    # Convert shape to the utm of the GEE image
    xy = np.array(shape.exterior.xy).T
    transformer = Transformer.from_crs("EPSG:4326", epsg)
    xy_utm = np.array(transformer.transform(xy[:, 1], xy[:, 0])).T
    shape_utm = geometry.Polygon(xy_utm)

    # Construct transform matrix
    west = shape_utm.bounds[0]
    north = shape_utm.bounds[-1]
    transform = Affine(30, 0, west, 0, -30, north)

    # rasterize the image
    img = rasterio.features.rasterize(
        [shape_utm],
        out_shape=[dims[1], dims[0]],
        transform=transform,
        fill=0
    )
    # full_coverage = np.sum(img) / (dims[0] * dims[1])
    return np.sum(img) / (dims[0] * dims[1])


def get_sentinel_date(data_id):
    exp = '_(\d{4})(\d{2})(\d{2})T'
    m = re.search(exp, data_id)
    year = int(m.group(1))
    month = int(m.group(2))
    day = int(m.group(3))

    date = datetime.date(year, month, day)

    return date.strftime('%Y-%m-%d')


def main(polygon_path, dataset, start, end, out):
    poly = get_polygon(polygon_path)
    lon, lat = poly.getInfo()['coordinates'][0][0]

    images = get_images(poly, start, end, dataset=dataset)
    image_list = images.toList(images.size())

    if dataset == 'sentinel':
        # Sentinel
        data = {
            'SPACECRAFT_NAME': [],
            'CLOUDY_PIXEL_OVER_LAND_PERCENTAGE': [],
            'CLOUD_COVERAGE_ASSESSMENT': [],
            'DATASTRIP_ID': [],
            'MEAN_SOLAR_AZIMUTH_ANGLE': [],
            'MEAN_SOLAR_ZENITH_ANGLE': [],
        }
    else:
        # Landsat
        data = {
            'LANDSAT_PRODUCT_ID': [],
            'DATE_ACQUIRED': [],
            'CLOUD_COVER': [],
            'CLOUD_COVER_LAND': [],
            'ORIENTATION': [],
            'SUN_AZIMUTH': [],
            'SUN_ELEVATION': [],
        }
    # urls = []
    # coverage = []
    dates = []
    print(f'Number of images: {images.size().getInfo()}')
    for i in range(images.size().getInfo()):
        print(f'Image: {i}')
        image = ee.Image(image_list.get(i)).clip(poly)
        props = image.getInfo()['properties']

        for name in data.keys():
            data[name].append(props[name])

        # if dataset == 'sentinel':
        #     data_id = props['DATASTRIP_ID']
        #     dates.append(get_sentinel_date(data_id))

        #     thumb_url = image.visualize(
        #         bands=['Red', 'Green', 'Blue'],
        #         min=0,
        #         max=65535,
        #         forceRgbOutput=True,
        #     ).getThumbURL({'scale': 10})
        # else:
        #     thumb_url = image.visualize(
        #         bands=['Swir1', 'Nir', 'Green'],
        #         forceRgbOutput=True,
        #     ).getThumbURL({'scale': 30})

        # urls.append(thumb_url)

        # if not i:
        #     band_info = image.getInfo()['bands'][0]
        #     epsg = band_info['crs']
        #     dims = band_info['dimensions']
        #     full_coverage = get_full_coverage(
        #         geometry.Polygon(poly.getInfo()['coordinates'][0]),
        #         epsg,
        #         dims
        #     )
        # coverage.append(get_coverage(image, full_coverage, poly))

    df = pandas.DataFrame(data=data)
    if len(dates):
        df['Date'] = dates
    # df['Area_cover_pct'] = coverage
    # df['Preview_URL'] = urls

    df.to_csv(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pull Image List')

    parser.add_argument('--poly', metavar='poly', type=str,
                        help='In path for the geopackage path')

    parser.add_argument('--dataset', metavar='dataset', type=str,
                        help='Which GEE dataset to use')

    parser.add_argument('--start', metavar='start', type=str,
                        help='start date to search for images')

    parser.add_argument('--end', metavar='end', type=str,
                        help='end date to search for images')

    parser.add_argument('--out', metavar='out', type=str,
                        help='outpath for the list')

    args = parser.parse_args()
    main(args.poly, args.dataset, args.start, args.end, args.out)

# root = '/home/greenberg/ExtraSpace/Misc/Port'
# polygon_path = 'Baltimore.gpkg'
# dataset = 'sentinel'
# start = '2022-03-01'
# end = '2023-03-01'
