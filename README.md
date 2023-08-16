# MultilingualBiasesEvalAndMitigate

<h3 align="center"> Challenges in Evaluating and Mitigating Gender Biases in Multilingual Settings </h3>

<h4 align="center"> Aniket Vashishtha*, Kabir Ahuja*, Sunayana Sitaram </h4>

<h5 align = "center"> <i>Published in Findings of ACL-2023</i> </h5>

While understanding and removing gender biases in language models has been a long-standing problem in Natural Language Processing, prior research work has primarily been limited to English. In this work, we investigate some of the challenges with evaluating and mitigating biases in multilingual settings which stem from a lack of existing benchmarks and resources for bias evaluation beyond English especially for non-western context. In this paper, we first create a benchmark for evaluating gender biases in pre-trained masked language models by extending DisCo to different Indian languages using human annotations. We extend various debiasing methods to work beyond English and evaluate their effectiveness for SOTA massively multilingual models on our proposed metric. Overall, our work highlights the challenges that arise while studying social biases in multilingual settings and provides resources as well as mitigation techniques to take a step toward scaling to more languages.

Paper: https://aclanthology.org/2023.findings-acl.21/

If you have any questions please contact [Aniket](mailto:aniketbbx@gmail.com), [Kabir](mailto:kabirahuja2431@gmail.com) or [Sunayana](mailto:sunayana.sitaram@microsoft.com).

#### Dependencies
- Compatible with Python3.7
- The necessary packages can be install through requirements.txt.

#### Setup

Finally, install the required packages by running:

```shell
pip install -r requirements.txt
```

#### Datasets

Download Multilingual Disco templates from [M-Disco benchmark](https://aka.ms/MultilingualDisco) and past in the `data/templates_corrected/` directory before running the `src.multilingual_disco.py` to get the final gender biases evaluation in all the languages being inspected.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
