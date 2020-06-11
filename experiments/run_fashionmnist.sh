






declare -a arr=("tshirt" "trouser" "pullover" "dress" "coat" "sandal" "shirt" "sneacker" "bag" "boot" )
for m in {0..2}
do
    echo "Manual Seed: $m"
    for i in "${arr[@]}";
    do
        echo "Running FashionMNIST. Anomaly Class: $i "
        python3 train.py --dataset fashionmnist --isize 32 --niter 15 --nc 1 --abnormal_class $i --manualseed $m 
    done
done
exit 0