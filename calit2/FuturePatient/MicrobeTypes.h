#ifndef MICROBE_TYPES_H
#define MICROBE_TYPES_H

enum SpecialMicrobeGraphType
{
    SMGT_AVERAGE=0,
    SMGT_HEALTHY_AVERAGE,
    SMGT_CROHNS_AVERAGE,
    SMGT_SMARR_AVERAGE,
    SMGT_SRS_AVERAGE,
    SMGT_SRX_AVERAGE
};

enum MicrobeGraphType
{
    MGT_SPECIES=0,
    MGT_FAMILY,
    MGT_GENUS,
    MGT_PHYLUM
};

#endif
