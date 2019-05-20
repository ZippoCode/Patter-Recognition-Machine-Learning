# Painting Reorganization

#### Estrattore della palette dei colori tramite la PCA

Questo algoritmo prende delle immagini di quadri famosi e crea delle nuove immagini organizzando le colonne dei pixel di volta in volta sulla base del loro primo principale componente, cioè nella direzione lungo la quale la varianza risulta essere massima.
Successivamente, dopo aver rioganizzato le colonne, prova a confrontale due per volta capovolgendone una delle due per verificare se esiste una maggior corrispondenza e in tal caso le salva in questo modo

#### Come lavora:

1. Carica l'immagine come matrice
2. Crea la nuova immagine utilizzando la PCA
3. Salva la nuova immagine

#### File

- input (cartella): contiene i quadri
- output (cartella): contiene le palette corrispettive ai quadri
- paint_reorganization.py 

#### Esempi

Eugène Delacroix - [La Libertà che guida il popolo](https://it.wikipedia.org/wiki/La_Libert%C3%A0_che_guida_il_popolo)

![Eugène Delacroix - La Libertà che guida il popolo.jpg](https://github.com/ZippoCode/pr_ml/blob/master/pca/painting_reorganize/esempi/Eug%C3%A8ne%20Delacroix%20-%20La%20Libert%C3%A0%20che%20guida%20il%20popolo.jpg?raw=true)

Vincent Van Gogh - [Notte stellata](https://it.wikipedia.org/wiki/Notte_stellata)

![Vincent Van Gogh - Notte stellata.jpg](https://github.com/ZippoCode/pr_ml/blob/master/pca/painting_reorganize/esempi/Vincent%20Van%20Gogh%20-%20Notte%20stellata.jpg?raw=true)

Salvator Dalí - [La persistenza della memoria](https://it.wikipedia.org/wiki/La_persistenza_della_memoria)

![Salvator Dalí - La persistenza della memoria.jgp](https://github.com/ZippoCode/pr_ml/blob/master/pca/painting_reorganize/esempi/Salvator%20Dal%C3%AD%20-%20La%20persistenza%20della%20memoria.jpg?raw=true)

##### Lista quadri:

* Leonardo Da Vinci - Ultima Cena
* Sandro Botticelli - Nascita di Venere
* Francesco Hayez - Il bacio
* Leonardo Da Vinci - Gioconda
* Caspar David Friedrich - Viandante sul mare di nebbia
* Vincent Van Gogh - I mangiatori di patate
* Eugène Delacroix - La Libertà che guida il popolo
* Hokusai - La grande onda di Kanagawa
* Georges Seurat - Una domenica pomeriggio sull'isola della Grande-Jatte
* Vincent Van Gogh - Notte stellata
* Leonardo Da Vinci - Creazione di Adamo
* Pablo Picasso - Guernica
* Giuseppe Pellizza da Volpedo - Il quarto stato
* Vasillj Kandinsky - Composizione VIII
* Gustav Klimt - Il Bacio
* René Magritte - Gli amanti
* Salvator Dalí - La persistenza della memoria
* Piet Mondrian - Composizione in rosso, blu e giallo
* František Kupka - Madame Kupla tra le verticali
* Pablo Picasso - Les demoiselles d'Avignon
* Rembrandt - Ronda di notte
* Edward Hopper - I nottambuli
* Jan Vermeer - Ragazza col turbante
