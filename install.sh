pes_path=$(pwd)


if !(grep -q ${pes_path} ~/.bashrc)
then
echo "export PATH=${pes_path}/pes:\${PATH}" >> ~/.bashrc
else
echo "pes already found on PATH. Please remove it from ~/.bashrc if you want to change the script location."
fi

chmod +x pes/pes

