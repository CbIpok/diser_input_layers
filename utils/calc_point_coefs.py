import nc
import restore

if __name__ == "__main__":
    marr = nc.load_mariogramm(r"E:\python\diser_input_layers\data\res\Tokai_most\functions.nc",724,200)
    fk = nc.load_fk(r"E:\python\diser_input_layers\data\res\Tokai_most\basis_4",724,200)
    print(restore.approximate_with_non_orthogonal_basis_orto(marr,fk))