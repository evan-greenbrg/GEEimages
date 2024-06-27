poly='Misc/Port/Baltimore.gpkg'
dataset='landsat'
start='2022-03-01'
end='2023-03-01'
out='test_landsat.csv'

# root='Misc/Port'
# poly='Misc/Port/Baltimore.gpkg'
# dataset='sentinel'
# id_file='file_list.txt'

python create_image_list.py --poly $poly --dataset $dataset --start $start --end $end --out $out
# python pull_image_list.py --root $root --poly $poly --dataset $dataset --id_file $id_file
