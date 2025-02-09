import pandas as pd
import csv
import argparse
from rdkit import Chem


def str_to_bool(s):
    if s.lower() in ['true', '1', 'yes']:
        return True
    elif s.lower() in ['false', '0', 'no']:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {s}")


def augment_smiles(smiles, num_augmentations, isomeric=False):
    """
    Generates `num_augmentations` augmented SMILES by randomizing the atom order.
    The first entry is always the canonical form.
    
    Args:
        smiles (str): The original SMILES string.
        num_augmentations (int): Number of augmented versions to generate.
    
    Returns:
        list: A list of `num_augmentations` augmented SMILES strings.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        
        augmented_smiles = [Chem.MolToSmiles(mol, isomericSmiles=isomeric, canonical=True)]  # First is canonical
        
        for _ in range(num_augmentations - 1):
            new_smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric, doRandom=True)
            augmented_smiles.append(new_smiles)
        
        return augmented_smiles
    
    except Exception as e:
        return [f"ERROR {str(e)}"] * num_augmentations


def augment_dataset(infile, outfile, num_augmentations, batch_size, isomeric=True,  train=False):
    """
    Augments the dataset by:
    1. Replicating each row `num_augmentations` times.
    2. Generating `num_augmentations` SMILES variations for each original SMILES.
    3. Keeping all other columns the same.
    4. Adding an `augid` column for tracking augmentations.

    Args:
        infile (str): Path to input CSV file.
        outfile (str): Path to save the augmented dataset.
        num_augmentations (int): Number of augmentations per molecule.
        batch_size (int): Number of molecules processed per batch (for efficiency).
    """
    
    # Load dataset
    df = pd.read_csv(infile)
    
    # Check if 'smiles' column exists
    if 'smiles' not in df.columns:
        raise ValueError("Error: 'smiles' column not found in dataset.")

    # Add `augid` column to track original molecules
    if not train:
        df.insert(0, 'augid', range(len(df)))

    # Prepare output file
    with open(outfile, 'w', newline='') as out_fh:
        writer = csv.writer(out_fh, quoting=csv.QUOTE_MINIMAL)
        
        # Write header
        writer.writerow(df.columns.tolist())  # No duplicate "augid"
        
        # Process dataset in chunks (batch processing)
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]  # Select batch
            
            rows_to_write = []
            for _, row in batch_df.iterrows():
                original_smiles = row['smiles']
                augmented_smiles_list = augment_smiles(original_smiles, num_augmentations, isomeric=isomeric)
                
                # Replicate row and replace the SMILES column
                for aug_idx, aug_smiles in enumerate(augmented_smiles_list):
                    new_row = row.copy()
                    new_row['smiles'] = aug_smiles
                    if not train:
                        new_row['augid'] = row['augid']  # Keep the same augid for each molecule
                    rows_to_write.append(new_row.values)  # Store in batch

            # Write batch to file
            writer.writerows(rows_to_write)

    print(f"Augmented dataset saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMILES Augmentation")
    parser.add_argument("--infile",  '-i', type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--outfile",  '-o', type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--naug", '-n', type=int, default=10, help="Number of augmentations per molecule")
    parser.add_argument("--batch_size", '-b', type=int, default=100, help="Batch size for processing")
    parser.add_argument('--isomeric', '-s', type=str, required=True, help="Use isomeric SMILES (True/False)")
    parser.add_argument('--train', '-t', type=str, required=False, default="False", help="Use isomeric SMILES (True/False)")
       

    args = parser.parse_args()
    ISOMERIC = str_to_bool(args.isomeric)  # Convert string to boolean
    TRAIN = str_to_bool(args.train)  # Convert string to boolean

    augment_dataset(args.infile, args.outfile, args.naug, args.batch_size, ISOMERIC, TRAIN)
