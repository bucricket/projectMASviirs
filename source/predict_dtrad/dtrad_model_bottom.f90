 endif
 endif
 endif

enddo
enddo

open(10,file='test.dat',form='unformatted',access='direct',recl=dx*dy*4)
write(10,rec=1) latn
close(10)

end program
