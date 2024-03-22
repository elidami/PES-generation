pes_path=$(pwd)


if !(grep -q ${pes_path} ~/.zshrc)
then
echo "export PATH=${pes_path}/pes:\${PATH}" >> ~/.zshrc
else
echo "pes already found on PATH. Please remove it from ~/.zshrc if you want to change the script location."
fi

chmod +x pes/pes

