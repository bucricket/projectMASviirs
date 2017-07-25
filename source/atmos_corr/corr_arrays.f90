module corr_arrays

use corr_parms

real :: lst(ilg,jlg)
real :: tsfc(ci,cj)
real :: qsfc(ci,cj)
real :: psfc(ci,cj)
real :: pres(kz)
real :: theta(ci,cj,kz)
real :: spfh(ci,cj,kz)
real :: prsmd(ilg,jlg)
real :: tc15(ilg,jlg)
real :: tc55(ilg,jlg)
real :: radini(ilg,jlg)
real :: radfin(ilg,jlg)
real :: satang(ilg,jlg)
integer :: lookup_i(ilg,jlg)
integer :: lookup_j(ilg,jlg)

end module
