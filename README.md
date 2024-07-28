# CAMP Llama2 server

This is an implementation of a Llama2 server using the [AMD RyzenAI-SW](https://github.com/amd/RyzenAI-SW) for the [CAMP](https://github.com/MHudomalj/CAMP) project. The server uses Llama2 model deployed on AMD Ryzen AI PC using the NPU. The server is build using FastAPI.

The server was tested and developed on Windows 11 with the RyzenAI-SW v1.1 version.

## Prerequisites:
1. Installation of AMD NPU driver, check [AMD instructions](https://ryzenai.docs.amd.com/en/latest/inst.html).
2. Repository of [AMD RyzenAI-SW](https://github.com/amd/RyzenAI-SW). Where the transformers example for Llama2 was set up. For default model use 3-bit quantization with lm_head and flash_attention:
```
python run_awq.py --w_bit 4 --task quantize --lm_head --flash_attention
```
3. When you successfully  quantized the model download the llama_server.py to RyzenAI-SW\example\transformers\model\llama2 folder.

## Depolying the server:
1. To run the server from CMD activate your conda environment:
```
conda activate ryzenai-transformers
```
2. And run setup.bat inside RyzenAI-SW\example\transformers folder:
```
setup.bat
```
3. Just the first time install fastapi and uvicorn inside your conda environment:
```
pip install fastapi uvicorn
```
4. Lastly, run the server inside RyzenAI-SW\example\transformers\model\llama2 folder:
```
python llama_server.py
```
By default the server will listen on localhost 127.0.0.1 port 3000 on endpoints /api/generate and /api/chat.
```
http://127.0.0.1:3000/api/generate
http://127.0.0.1:3000/api/chat
```

## Optional arguments:
```
  -h, --help            show this help message and exit
  --host HOST           IP address of the server. default=127.0.0.1
  --port PORT           PORT number of the server. default=3000
  --w_bit {3,4}         Quantized bit size. default=3
  --flash_attention     Enable flash attention. default=store_true
  --lm_head             Enable quantization of lm_head layer.
                        default=store_true
  --num_torch_threads {1,2,3,4,5,6,7,8}
                        Number of torch threads. default=8
```

## Testing
In another terminal you can run the llama_test.py that sends requests to test the serve. It requires python package:
```
pip install requests
```

## Help with Llama2 model download:
Request the access on [Meta webpage](https://llama.meta.com/llama-downloads/). Under previous models select Llama2 and provide the required information (you can unselect the newer models). On your mail you will receive a web link that you will need for download. For download you use [download.sh](https://github.com/meta-llama/llama/blob/main/download.sh), which you put inside RyzenAI-SW\example\transformers\model\llama2\llama-2-wts folder. For the script to run it is best to have Windows Subsystem for Linux (WSL). To enable this feature in Windows just type WSL in search bar and follow the installation process. Run it with the following command. You will be asked for the link from the mail and then select only 7B model. The download process will take some time.
```
bash download.sh
```