# Notes on with the fune-tuned 7b-chat_ATT0



<s>Abstract :
The total structure determination of thiol-protected Au clusters has long been a major issue in cluster research. Herein, we report an unusual single crystal structure of a 25-gold-atom cluster (1.27 nm diameter, surface-to-surface distance) protected by eighteen phenylethanethiol ligands. The Au25 cluster features a centered icosahedral Au13 core capped by twelve gold atoms that are situated in six pairs around the three mutually perpendicular 2-fold axes of the icosahedron. The thiolate ligands bind to the Au25 core in an exclusive bridging mode. This highly symmetric structure is distinctly different from recent predictions of density functional theory, and it also violates the empirical golden rule""cluster of clusters"", which would predict a biicosahedral structure via vertex sharing of two icosahedral M13 building blocks as previously established in various 25-atom metal clusters protected by phosphine ligands. These results point to the importance of the ligand-gold core interactions. The Au25(SR)18 clusters exhibit multiple molecular-like absorption bands, and we find the results are in good correspondence with time-dependent density functional theory calculations for the observed structure. Copyright © 2008 American Chemical Society.

Title : Correlating the crystal structure of A thiol-protected Au25 cluster and optical properties</s>

### Testing text generations


Input - 
<s>Abstract :
The total structure determination of thiol-protected Au clusters has long been a major issue in cluster research. Herein, we report an unusual single crystal structure of a 25-gold-atom cluster (1.27 nm diameter, surface-to-surface distance) protected by eighteen phenylethanethiol ligands. The Au25 cluster features a centered icosahedral Au13 core capped by twelve gold atoms that are situated in six pairs around the three mutually perpendicular 2-fold axes of the icosahedron. The thiolate ligands bind to the Au25 core in an exclusive bridging mode. This highly symmetric structure is distinctly different from recent predictions of density functional theory, and it also violates the empirical golden rule""cluster of clusters"", which would predict a biicosahedral structure via vertex sharing of two icosahedral M13 building blocks as previously established in various 25-atom metal clusters protected by phosphine ligands. These results point to the importance of the ligand-gold core interactions. The Au25(SR)18 clusters exhibit multiple molecular-like absorption bands, and we find the results are in good correspondence with time-dependent density functional theory calculations for the observed structure. Copyright © 2008 American Chemical Society.

Output-Title : Correlating the crystal structure of A thiol-protected Au25 cluster and optical properties</s>
```python
pipe = pipeline(task="text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                
                # stopping_criteria=stopping_criteria, # Without this the model will ramble.
                temperature=0.0, # randomsness of out puts 0.0 min 1.0 max
                max_length=512, # Max number of tokens to generate in the output
                repetition_penalty=1.1 # Without this output begins repeating
                )
```

- temperature=0.25,max_length=512,repetition_penalty=1.2


"[/INST] Title : Single Crystallization of Gold Icosahedra: A New Structure for Thiol-Protected Metal Clusters and Implications for Density Functional Theory Predictions of Cluster Structures Based on Empirical Rules. © 2008 American Chemical Society.

[/INST] Importance : We present here an important new structure of a 25 atom gold cluster protected by 18 phenylthiols, which challenges previous assumptions about the nature of these types of structures. Our experimental data show that this cluster consists of a central icosahedric Au13 core surrounded by 12 additional gold atoms arranged in six pairs along the three mutualy perpendicular 2-fold axis of the icosahedron. In addition, our data demonstrate that the phenylethanethiol ligands bind exclusively through"


- temperature=0.75,max_length=512,repetition_penalty=1.1

"""[/INST] Title : Structures of 25-Gold-Atom Clusters Protected by Phenylethanethiols: Unusual Symmetry Revealed through X-ray Crystallography Methods and Combinations of Ab initio Computational Approaches and Experimental Results Supporting Them Density Functional Theory Predictions Are Not Validated by Empirical Data for Such Systems. Composition and Structure Prediction of Small Gold Nucleic Acid Analogues Based on the Presented Highly Symmetric Structure of Au25(SR) 18 Particles Asymptotic Spectra of Au25(SR)18 Versus Spherical C60 Fullerenes and Their Electronic Band Gaps Calculation on the Basis of Time-Dependent Density Functional Theory."""




- temperature=0.50,max_length=512,repetition_penalty=1.1
"""[/INST] Title : Single Crystals of Gold Clusters Protected by Phenylthiols: A Unusual Octahedrial Structure and Empirical Golden Rule Violation for Metal Nanoclusters Revealed by X-ray Diffraction and Theory Calculations  © 2008 American Chemical Society.

[/INST] Author : Zhang, Shan; Liu, Jing; Li, Feng; Chen, Hua; Cui, Baoqu; Gao, Fei; Zhou, Hong; Zhu, Haiyan

[/INST] Journal : Inorganic chemistry

[/INST] Volume : 47

[/INST] Issue : 9

[/INST] Pages : 3692–3695

[/INST] Year : 2008"""


- temperature=0.10,max_length=512,repetition_penalty=1.1
"""[/INST] Title : Structure of Au25(SR)18: A Single Crystal Cluster Protected by Eighteen Phenylethanethiols and Its Empirical Rules for Symmetry and Ligand Interactions in Gold Nanoclusters  © 2008 American Chemical Society.

[/INST] Keywords : gold nanoparticles; ligand field effects; protein engineering; self-assembly; supramolecular chemistry; synthesis and characterization; theoretical methods; X-ray diffraction"""


- temperature=0.0,max_length=512,repetition_penalty=1.1
"""[/INST] Title : Structural characterization of a unique icosahedrally coordinated 25-Au atom cluster: A new "cluster of clusters" structure? Journal of Physical Chemistry B 2009; 113(46):15398–15405. doi: 10.1021/jzb00010x

[/INST] Keywords : Gold clusters; Icosahedra; Phenylethylthiols; Protecting groups; Thiolate; Time-dependent density functional theory"""