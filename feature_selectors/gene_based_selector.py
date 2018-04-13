from feature_selectors.base_selector import BaseSelector


class GeneIndexSelector(BaseSelector):
    def select_features(self, data, labels):
        # Select the regions with genes ERBB2 (HER2 gene), ESR1 (ER gene) and NR3C3 (PR gene)
        # According to :
        # http://may2009.archive.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g=ENSG00000141736
        # http://may2009.archive.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g=ENSG00000091831
        # http://may2009.archive.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g=ENSG00000082175

        gene_list = [876, 877, 878, 1577, 2184]
        return gene_list