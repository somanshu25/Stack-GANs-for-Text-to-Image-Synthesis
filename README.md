# Stack-GANs-for-Text-to-Image-Synthesis
- [PyTorch Code to run the experiment](https://github.com/somanshu25/Stack-GANs-for-Text-to-Image-Synthesis/tree/master/code)
### Data
1. Download the text embeddings from [Text Embeddings](https://drive.google.com/file/d/0B3y_msrWZaXLT1BZdVdycDY5TEE/view) </br>
2. Download the birds dataset with annotations from [CUB Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) </br>
3. Create a directory called `Data` in the root of the repository. </br>
4. Make another folder called `birds` inside the `Data` folder. </br>
5. Store the downloaded text embeddings and annotated dataset inside the `birds` folder </br>
6. Any pretrained models related to Stage-I must be stored in `output/birds_final/Model/` </br>
7. Any pretrained models related to Stage -II must be stored in `output/birds_final/eval/` </br>
### Setting up Environment
 Download the environmnent.yml file and run
`conda env create -f environment.yml`
### Run Experiment
1. To train Stage-I use the following command </br>
`python code/init_stage1.py` </br>
2. To run Stage-II using Stage-I trained models, use the following command </br>
`python code/init_stage2.py` </br>
### Testing Experiment
1. To run on unseen dataset after getting the final Stage-II model, run the following command </br>
`python code/init_eval.py`
