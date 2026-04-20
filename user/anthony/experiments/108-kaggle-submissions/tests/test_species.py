from birdclef_kaggle.species import NUM_SPECIES, build_species_list, load_species_mapping


def test_build_species_list_shape():
    species = build_species_list()
    assert len(species) == NUM_SPECIES == 234


def test_build_species_list_sorted():
    species = build_species_list()
    assert species == sorted(species)
    assert len(set(species)) == len(species)


def test_load_species_mapping_keys():
    mapping = load_species_mapping()
    assert "birdclef_species" in mapping
    assert len(mapping["birdclef_species"]) == NUM_SPECIES
