"""Sparse closed-loop route agent for Kaggriculture -- vendored, not ours.

Source: Kaito Fukami, "23/23 Strict-Future | v23 Sparse Closed Loop", a public
Kaggle notebook for this competition.
https://www.kaggle.com/code/kaitofukami/23-23-strict-future-v23-sparse-closed-loop
Version 1, dated 2026-08-07T09:03, public leaderboard 2492.6.

That number is lower than v22's 2957.6 and v21.1's 2665.8, and it is *not*
evidence that this is the weaker agent: public scores in this competition decay
as the field strengthens, and this one was recorded a day after both. Only the
local gate settles it, which is the whole reason this file exists beside the
other two rather than instead of them.

The notebook ships the agent base85-encoded, zlib-compressed and SHA-256
checked rather than as readable source; this file is that blob decoded, byte
for byte, with only this header added. The blob is 29,999 bytes and hashes to
76f6fce5470b568410d79aa1ea7b32ac1f0616b6584d6cd6e146da1cbb037252, which is the
checksum the notebook itself asserts. Attribution matters more than usual here:
the author chose not to publish it in readable form, and nothing below this
docstring is our work.

Unlike v21.1 and v22 its `agent` takes the engine configuration as a second
argument, and branches on it: `townCenterSellInterval >= 24` selects a route
fitted for the announced PR #1394 rebalance, and anything below it selects the
legacy route. **That branch has flipped.** kaggle-environments 1.32.6, which
the ladder moved to on 2026-08-07 and which this project installed on
2026-08-08, reports 24 -- so this file now plays the rebalance table, which had
never once been executed in this repository before. Every earlier measurement
of "v23" here measured a different agent than the one that runs today, and none
of them transfers. It is the only vendored agent that adapts on its own; v21.1
and v22 carry one table each and cannot.

The author's own measurement, from the notebook's first cell: the legacy branch
wins 42 of 45 at a mean margin of +3,249, the route refresh alone wins 40, and
"stale public v22" wins 3 of 23 at -3,228. Those are all legacy-branch numbers,
so they describe the branch we no longer run.

Left unlinted and untyped for the same reason `kaito_policy.py` is: reformatting
code we did not write and cannot meaningfully review would only obscure which
lines are ours. Its behaviour is pinned by `tests/test_vendored_policies.py`.

The author's own description of it, verbatim:

    v23 sparse closed-loop agent for Kaggriculture.

    The farm plan is selected once from the official engine configuration:
    the strict-future Azelearn route for the legacy engine and the seed-robust
    Khanh route for the announced PR #1394 regime.  Runtime feedback is limited
    to actor-local weed recovery and ordering existing SELL slots by the official
    price curve plus a bounded demand-recovery term in the rebalance regime.
"""

import base64
import copy
import json
import math
import zlib

_LEGACY_ACTIONS = json.loads(
    zlib.decompress(
        base64.b85decode(
            (
                "c-"
                "rk<O>bM*5&bV(a}iRMY<H*FOe{pP3`s7L8bT1DDGC(nBJHl|f3IRu<l~!}GiPS*eWX;cOp_Gf{k}72&Yb!DpOe4+^80VU{(kbO&n"
                "F)*Zf{TSXD5IE<)453*TWkRFMs>x_kaBQ?}yi)Pu^eNtR6n)UVQuMr=KrAT>f}*buv47d%ZrHEyUZm?^mmjgAZ<3t9K7?uivjO?o"
                "MW}MsNSPy1M>wGMjI|{_*De-"
                "KV#AyT3TQzyI%SuRoV>fB5uq_oVrtZ=X-rtJ}LTZT)b4bNBwstL<B(7yAQoySlpCJ#}tAb^C?kQ@5WE2j%kW-"
                "Omq$fA?uKdR#lyK@iQ^{Djto-LTk?%#8tf=;lvO`}chO>x0(prd*l)@ms@ZkLUX8;_YfXl6#MkebXK)UV(>wH{2ilgS%<QA8YE5-"
                "+K7}hr8{b(cg(Y`R&Dc07q?k7^;iA)y?SD(?fUP8Z`rr?C31oHe%TF)y1vx^w2Nweozk6_7U3`x7VMX;gU~K7W%fu+aGQ_T;ny-e"
                "AL7eknLBV`S>C?@w@h-"
                "8LJGMJbng;L20x$tA?58+3?$m@j{DDZq7CXH$DhEj3BYz<U8OR$*emxe3^4$^Uk21yLYTd<sP66*7l>(Ba=PYwO9P`<BP!WqOSt;"
                "3jEpADrCNEU9^E45`Fdh>T31&?&t4TH+Pp;mw)*(YOT94MSI3Z9{B3hdG@vG<)%lr%Ga+#k9On445nbVwB6Z&zq$Fs68f8wogVsY"
                "+fS(3{P3??rwp(An2peDm?CYE!_>fYZ8=EN73XatarVU?ZIAHIzIEH63@~a!t0CSwNnQh$0S=6jGQ<H-v-"
                "=vB?nkQu9xQ=`**SKflV0A`PcE3+k*h0lFU5}82A5Fg>gll#bGLEf?8~2cU9OV!@fN%{yyv(_S>gcec=v`z%s<HOn>ICM?$I>q8V"
                "kGsFX>ORF30Fa$<<)uDdXr{+}57;@={_yOgJvi4|X%tXU)7o8(l5jNe0CJ{^I7(@jGj5;;X3rrYUKhL@^_SV}h*r-"
                "EWTr8JWA!0Fq17CYRNAveFZ$;q@-mUYJ4cj4JuJS_gP{4*I;>YAaYa!^ztH;NH9O`IB=6B8<Lek~-u{fu60CBxA44il$-"
                "|=F(0^SDs!dGg|ybD{xaaK{l7#;_S;Z@XQivA<X-GnH8TQeeNBNk2$rE{qFMnPUb(o!y5UT72J$C5G_-"
                "uBT5z;v&RO;hD>Wmj*8U6rD8|#CHsw2xth-{hE?(er||kB-iLkcA2fsotmWefVFf}Y#TeGkpi4#2s1R5N-"
                "vsVQTWGlKMf=N`aGAq1)WSV$HANlmgE5yg8Oo}0*V@Fg*4Nh$|IEKZ{>UGL_FzrEE2JN6yqMd&n~U|gtDBpj9-"
                "jHe<ajLZGcKEA(2Tq;Hjg54M=Kcy5;l~swR}OJEDHiG9=F+Om*lR>b_MwPycwb6lnQ}4@O0n)z+DeMy`jHmFvqzOdh+)MTP$rd3e"
                "vCkWG>YrawFO6M#(}54}mhFsd)r;n}UU7q#Ru&y%v_%B9Ibf3~GA0Nuvd1wd~p|nf#+C=WtNXDZdz>fpDCHaSL`<Ls28^770&Z1J"
                ";r=UV+)`Xuv5+2HIh7#t)O`x!NpsMvY-"
                "|cY*PCDSoEJ6y1hu^XWDs;8+{`P5Trb`hg@hX*I#IZn5?9fE>vF={9@F(~*fbFkDz;DO#=pm=f=E^k=Qwrlj(yQ+6OK40iP90bfq"
                "~B*0@?Cenqqz}fjd&(PlY<>uKmW&1djNp&*4Dx0q?FKB7dkmfl%qwa01UwcKy5uzaA^RMEG3`5Oh%UVgJPuNppaY*|Zuot>oOWPO"
                "7F82eDn{^m4j8q0l4d>VLOjm5Ww4Al!y5P+`VtqJs!z_+<gqq}LuFNR15_-"
                "bsZ&b`z+rtrafN(FJW5OLE*?dSf<4I5y^>FV1qEmh0RuC4gnXVig7JLfxKdr_Z6sBY@3tEFsfdyF^kS4!N*>^u)Uj6VS+cQ#)d1X"
                "NT=V3E2hL_KMWZG1&h*^GyX*y<7&Di@<-"
                "dEC8)XmkJ&J8*bs8h{uND&W)cr0nR){(HCU1D6g3`w#z8tfXoW>jG+e!RJaAag!qKFbZWXC7UcK_;8No*4;hCG$%sXM`%(0Mb-"
                "aK-ArHOIAK@b!XQ_$pYN=Tk$b8=_KC6lKtL2i=TY64ashT;4qaw#CH-"
                "b+a9L8WR2^NT+X~liQ8b^60l>I%p^~<?ZdusduLVN)b{6$)heH>2r{*qTWPHh+<ysLmS{4(tz3gggHOua>61MO@O)lNf8exg_Otf"
                "?0`O(#oFZsQsxm2rnovCTbvG9|UxDdtBEd80J0x0x^4mz?!DhYmfT&r!kOG$Labh$?YOY}<iL`;^r4NqY8b%hC`paY@EQOGQ);h^"
                "$y=Xd!qOfL0nGp~!Kp?aywE_+eCyH9|_mrKq#+xUwjnbnDNHnU|j2C5i^E6!t;95aZqTp%JsQKCo(U+~V{jeiti5TZvzPQiC2DAZ"
                "3$b#R(1wXbhtfP(HyPNBe94%UhxQB*XdR%FemRQ}=??Qv#cBlb&!FD9J<Gp<$<_8V*Vi_6e(>85J9_M2>qHE){GRy-lja-"
                "}kB!$3`LR=!$!0n0v?Q1ipYAlb5;6<I}!nn5e+^N<DMv#mDDU<aW6FdjIm`U<sz5f<tSSW?Uveyh0V3?@1IBh^-"
                "gLsGXp`U;=b11q7z)Tp{DX!eGtjYMzh55}`H!Zp<AgB~b$1Y26slB-{-&DGY;T9R~06du@B~X?Uz?mx!Plp2j4@o%`lM8a-"
                "09}&d<ZD)NWcoOw<914wq7d=<^BCeRal9?=<WnQ<U=T}*huIEoxPp>TDWjiMa!+9Cnb;9AL(4|H&v=`vVqzmj%xVm-VxEMI@Q~19"
                "gV8aB@lsTUBHza*0D-hs%)VjobR~{mkjEzWBA7NfNnYMhb~P%<F1>RXeRr-B1rM&v8K=`&%DhI<nG!q)`8bG(61e<Fz(;Ivs-"
                "S8pR=%2Qoo2eH)tegK4oPM}c*T_%zOo;YR?~-ylJJXM=$gRS-"
                "npLwLX-`ZNxAFQyX20N-E%?(O*kau>XWq=d+W;_nq2E^+r!J(?6A5f+ajSf(54%q9oklc%WMEe0nCO-"
                "OgcA|0oIj+R%?SzHfW<T2i#iQMlVrmA~v+$1dkk$Y1wHj<JM_pIGHpbU-FpAItz$OAIL8;J7FLKvg8mFHE)0-"
                "bxeqBuvD19`YNgSiUGqJk43Ox?WDbm(eb$}Z89b8V+n}0*U~?op@6k*OGV`rpv<F-"
                "^K4Eo<5lr*DbO*1cH8JK1kDn~2ek3j&ZK0LGRl?pCNO|XCDmFioK{%4YTUl0YDw7ondIc$6I*+1|5xb+!_;DqhVbeIFb@vn`NXoy"
                "P#I9oJzCxe#KLxf0&FPC_#~9rYT~1Exa>G0EU1*p)FT96D4!19yG{@$tuRXq#e^L}sG%7g@aELdjDv{P=^D!&5^FwAh+tU{_6bwl"
                "1-"
                "7w=q<h$$e}1$l+yzj267T>D0YS1VNwx9pwB$p(=u&FE_+wL%lRpZrdiIRT53gBXRSyG)C9w1(1v*uQ!~|_p`I|{kagR!jWSo0M_y"
                "qOlTqXMKa@8i!#t)&6!-}AJ1bkRK(j;MVxt%=^c-^+zoD3P^97Q+NpxPd``)UO~-=l8tsjqwugz24aC0itrpD`VrF$F-"
                "d2L~vbMCXT5ADATO*vx%qqKi!(<WNkpP#YuS6L*a%wlwwws<(ZjiJl>EL=$Zfd|KZ$!1j(9S`pV%h8($?;>{w~(bX03&C-"
                "=eD+>)nB_QhAAT>4<pX3Az+D0c$>v($SM;`Yr2)T&&ATuJ^Fs5(%D)Zt!M)9)-"
                "3CJ6LuawU(na3cgIi%ZH>rz8wFRqW+g@{>>*c`eAUmx@`<D8}&Gl*(-"
                "5mTEYCH<uaHJh<Jb<2QK@@x^nj06gStxTlk79*M@eGkW#8U-<rgyaX8Y;F7*bBgjhyM)gw>Ah*k5W5-"
                "8W0%rQrX0uigWYPe6V*)mhcF{7y<m}#_=0aF5*%5&%307P2F&<~o?islclZ?@mLxbF%CZY4wXTSK`&6kqNc;2|teI@anz{rOmZKC"
                "3+5{Y_FrylRQbK!DUp}zQ6{%kt?P0GiB$*9jTy1wcidM<Eq>`8l)XoZ40&sY=hI#?ZSQNDsrc7$Yb@Gvqt2Lz%<fQavrGS$3{g2k"
                "c57L*9CU&a)g{V}jesYdKX_p7uFjA{F_tbDp_ynbH@m-|VhKej7OB5Kj{~5=M_vdr-_xr1=SunSKjCnx8H=E9o!;dXAz2=m~haRj"
                "qbyLQ)l`R4qn2~)uSS%c~7hzrUQaOc$_Bgx$!PGF@3VespL&n8+eVB=hy6?Tpu1F$WubofsmIvENISxu5F;sw*m-"
                "x_a?zL1hslCdfR7V@LD}}0#jM&gLRe>x9sh|Rt8r$+z!OEmcL%J^!=?2AfH^9csb}kUUqr6{<o)|?u03SRM@noD!s*K!9a#m^!qD"
                "9Mdib+hvTD7pM1cff>Y55a}tzG@<Mbayd1%_c>SWv$Qtt>j^kW(`F#8n%uJP)k9;VF)Vq}#O5_F#39lHt36Zb($7PBxfuR0HC}*l"
                "i06*mp&Iloy_N%4PjoE>wY(&VxK%Y<0-"
                "f5W~|52&Jl+tqIki*p$R~2i(E&V7z_6_Y|3~>jiRRQ*e_UVYYm<FhRmEY?~$z1|KP?I0gPf8ORMcgG*Qn`P<q_POoqKz|(g@c|`d"
                "feiEG1UMYp=aB0n9874-C$QTg7-$CIaRL`bVN8OcL12h+9_?dgM%(at1f!#EOYN0e~zNiW!vZ$nER0b1a1f-"
                "Bf5)k6@=mFOTv2RLH714+onPa^7-"
                "TglQrl#CpFDi&^p*+iEv4moHLo&dd!TZCZIX6l$9t8h*rc!yDC&*}rX3>qTfOxsVeYM58=JgQ?Xd$#kHv5=W$#Imp)R}&g<;vx&L"
                "sk(kb4Ju0I@Bn1$C00Mxl&nYhSXp0ymZ6ck#iBVGVh@x_H88kbR4)6y>(p%wln~fIFKxOr*@x`t9EAFbq64YgSp0FT7y2D{$___`"
                "79uuEuNO8>JX`4NYE{_ZzB$~;^A^BDMo<30-Ql{2;?s*?F=}S@(yt+C<^c*y&Q*ur6p<Yc`<JcFD87gD4xd1RGuqE>piQ-NlT#|v"
                "Q(*fP@_$|O%M~L80S}s_KZ6}TFk{})QW!<#>B@QECu`5(&OF}vD@X3JaYm2V<961D%yu|`aJJIH>*G>E$UFV%;R99n9W-"
                "(FDBikleP~uZ4U3E!zgT;eGY~n6BZ~VYGv|2sg!e?CEt#SApA6yedEC;IXbA;H5gVYIiUH(6K?C^@}rn_p9=U#(od2Vrl}(rc;YF"
                "@F-"
                ">H^iY|)zkHY+^Nt!p{E~IiENQbq}3?=9cAWZ_gnV42mkg%ZW4cTpjj{+q$@68JnnMY&@hOYy29Ha9Pfr=DA8KMhuCL11A?0qg$mF"
                ")x4V9JdL@fHOb10%l~Ty<4|>_k+6f#gZm27SivWK@s9ms+LEHKNUd2rxOXT@9lvPu2(s9Lb6?XG1=*T2Pfqc#+vQo!XUD>F+mWRb"
                "h@XuGkWkIpqlx^KV{~EXUEIGqz%dE!QFG4q@F6oq(tP(HYO#n4k76Ye?}6QGp-"
                "+2&&|$gd{PVwYZbQDT59Lh!g8oi>N+`h3~%N=TPPpBh0WcjAC}ymQ1F4!`Mzl2>`|a!a5xN$`Y`ADil;CDy;{?#({0?wm`$%bg-B"
                "gr(g=ZGah1=qoce*Y|zjOWMnj}DfsP4KS=MH42mO&yfGI@p=jhhHYUA1Fl8P9-"
                "=KI3Bu4SaR6s}}tj!Tu!0Etga!4+4O{+hcqdP(#PtI~`wPsPeMMN?eYx5{MONb{)so+2ic>nx2f1{LRD6b#s|0flV5z-"
                "2yU5XS)f<mQwDi-K6(@W$;<_&y%P{ij6^0nbb9W`8CU4Qr@Q_(;OO9NLGY=&HZ+X3A5Z6zz-JA1d>X{HvdWJFol8LiZ_Vg!|PDUz"
                "bGN)R@=RAv7aun>o4GR1K{x*!7jTu~Z4n;P_pESNo}m4*mbkUNA0qcp)akZfw5J+|3<3&}g%lgg0*X|clUS9V&Aod#0jtsrq+hB#"
                "5Mh>#HDGF}?$(L{7qL(&g5+8XA230o73&p3%0)8v>n(oSW?bsQ(B<gvFYKk7M@y3vLD)r{fhQKkG%1C;v=ml|VqjJ=V`y^FeSpkE"
                "-1p^X!!*EB2oVUQt(7hV@(9&*8v>^7{snPTriz86nh<5;+o6O|%I%?e~TzCv$a`>FBP(RhZ^Pm1jvM^{fh%0T2Z=J&<xL?Pc5#h#"
                "R<2bBm^X$U@JTSFZ%-A(Rsg+$30E|g<!p{%)*e|#-"
                "4)4Gg#*o5AQmU#udNUo|<la@gpU96@O7I%5=yOqj5mk)N>X#jSRetKqH<rSd}u-"
                "%9tQ_W9YJT(+!76CV&0z+b?w(U<oj#pdlpO8EfsS3m=l%=s7nKUc0En!F~oVR5b>Zt=e?f7iZLLSJbzac@<An)z$AX>_>-"
                "zYL;^8NxBX(ZEe?Q69JyO6p1L#3*0W(209LE%}d)kh%M8g)jr?uQBp<SWOPg_v}(3$iDvgv)7D3_T$kQh*4q<4lolj%1dr%7YPHK"
                "aM<Yav`C{#_l`Au}-"
                "YO78@)s6NOTj){#)T^_@lt$hu)Qb&dm=D}$7k<?$ePEHcAM?^dcbd+h_q1_$OuhdZ+FA_SA+(|xl8@zb;&)n+BZ4$vjRa4afDzsM"
                "{yQgh5>iP9mbV8yD!0&{RV<wk{Zj4bW0(Q1?IP8AFQlG_{)PQgYT11cdZPlze0fhuQh#ppf$mkRW9MS|No2mnmMaFk;8wn}>6)jD"
                "9*mohMB?~-85oOCD~wg)LHIN9QBw&=8MDXJ7EctNIip7*RaMpj$Mb*Eu&XSBgwq4a4>l10UsDmvs-"
                ")?(7T|L*epP_AlUE=avdlb`&DLf{^*HZ3%f5n+FtPu~#JRzbFC6-P3w6n)Zz!eT15L*}V-"
                "Pyfoldio?`9NP#X*qtlV55s*8B*;{!Q_AS*)HK?CMp2Td*}%o)FkI*`_fG=?J5mNX!Cu3cpCfIEmN^Ewa;lAz-"
                "L|D1fZI~iha9PH?*LkR@<|r);Ebog5g`kc?Y=OsT+TA%Po<iipfD+50L(*<24p&k?6o|uItDpL#E?+cIf95QFCG#z2RU%|ITis3p"
                "+Gq*jfTigklDvmT9k$>KSR*aAr2O~aCynnkR-8Z^-"
                ";CzWU}1&I1#rgSY3ONML>Hl0jsfSnxeBW8C1*>W1mNA3I)S%We^uYA4K`5fSqY;@fA@w<p@BZG0m5=*LAQ7ZGV-5r}Pfhsu*>T*-"
                "B*@x;dokNi(R4Fx^pRUk+bzcEHs1JtTPm73AjrSC%{y`KDuGKuX6S`F2ehdMVyp*QOt$o*>r`D4Lg`+#IshpejWKllN@y;vowW;v"
                "U)QD1@?CBqZ0oav`39m`7+EK3yvvLUI9IY7XE!srVVsi}m#JC`2(wkD@itJa`_rjIp50>MY6auYdq-XDHG{WmckF9&v^cfC=^1lG"
                "CAVW3DJWTPij{Dqa(@R5wKx4YQk5pHX{E4!k09h0{D^=L@RD0%i6ibrsFn{KD+cDpK#!<A-"
                "M^Kxw@K%xe>X)Jq*D#6XImsh#etK_X&+TA~q^SV3%Vv+PGG9;oa9&YCh{4IPTGuMiSGh9y4W_K*}gH+n!Rcq_|p6j5@~lCwew)v5"
                "6Mql3*Tr5V<i$?SYo`V&aL1uI#Mdfcu2<>NwCgv1sW$n{lWtQ~+ALbBlXD^aP2^f=Wn@QVX9hoLfMk4Rt!Q`Rz>Y;09bj-"
                "o*%haP!NOql!$J%ZRwRBK}5EJ2^%*aG)DNKH&c)5E?(O*t%H$`4Nw7J6%}8Cb4(iRnaa3DyeO71|xhEyTmkmu8SQetqe)%IpCaXQ"
                "&76Itf+8V>jm5OU*SoxyDhQ!8jh_?^f9ZAngWmXGG>VCxfKTK{pB1slWv%QQM(=5!LKDzwcF^z!R@5U?YB$T{szCLTc>A=FHGU_k"
                "b@tkZN-XKb}$JD|QcY0$w3;zUY-lg_u7KcG;;}qy|+0*lyJ_2AnuRvnY+Rm|SfjrSJ`tQMiYsE+lC3_~ozQNb~ZO3#`mV>v)~Pw8"
                "TSba*>Rsr3Vn$0%We~*ChWCAr-!m{`7SugXMC-ZQreABof09*5QKHn2xj*WU%&DQ`%UYN{){7F4!@#`jnHDaHOOnD0KzEC7YsPBT"
                "&F{aimO1)oudq^2E?hdJ<-"
                "haTM}dxSCUy1QwczfAP6N<k=zpwAKX{)5Swg1R%zz!DS4*BD4XW_Y&2L;VT7@L$;(gVxMHf7s?O;v@oc{#NbO=xwxbpUaVSbXnU7"
                "vScuh+XfDd(qTuP63!0h<vIGEA;<73gmE{*pM_9S*V2~0ffO=qAj18W{NSF$W3j7h0qHmqSCMiiVeJzC4#Ym~K)`q3Uc-XOuLT!r"
                "ty%x!4w${}*5`bF1)C9CcB2tJ##=Mhd@GLsj2&DRh<h<Ilda^|W57Lct89^A$C6%zit7EP!=m~fwbikZ&z_ir{5rKvR6ZZN1ia^V"
                "k_o7H{z0x40bMP5=w%`h1UA#>JLWilc<3kHQkzm3>9nZ@0iu~9)j!X+#izagMJ!=!<zENe{n2brB{Zq_KN>%;ja31Z$%o&3KJw?_"
                "#sMdsH9;x$#Bi;bgsHa)POBu1J&N_L(q;)G6jHnS&TPJe*Ft2UkI-9J8<WlkZvGcbhAMI-ah4~AB(q<cFQ6Y1@7(sWvN}|--"
                "I}LE)(qRI?dvo}pin(6UuB7VF;^LgJ7>7dfF{UpuQ_K5YiHgBm(y|A$hXx*rfc`$4x}j)A6^{!-go~ANI*VFbo-"
                "i9(^I4k(hN73D^MZjxm0fI{N)ERxvE}m|mt_sAePB=A<xg(>g1+Bh9}En1EwQubE_To3e#wI8X&%{{PuX*)>D7IgYL1A|M(Lv;x_"
                "V={kw#Mf?3ju;vfYDGrRPwg(avEyhGL9($Q?Fj0)9%CUW{i%1`+oT6%--2&?gx-"
                "$t|G^B6_f@x)KnzIpFMP3mMWS4Gn)Rv~W?q8O&A2p9aPd7^`e4La3b*l;d|>=MvKFy#tFj;sJg9jTV`o4Iu$FPsem-"
                "LJJf5j+zpgouFe%28XB-N_(^;IhQfirab|@@mS(eqIr}LIbCzegtF-"
                "x)Vqw_b8QS;xNgq{P(1wQ{P|X{3*11CsmM@m68CdNc$ll8mV7+heOw)3l0j%Y12A<bFaQxyfG&EAlSCj5!f%C2)1ZZILq+k3A&Ju"
                "aVa3Ok&WdaT&EO?vO(~Z+FfFQUY|LO*xCLbbL`taBqEUh8V`Wh<0VLwP$U<m4kwP~{<fgYd(o#MOA1R-"
                "~7Yj|Q6*A98N+C?nZ}IU39%@CofrT`gjLlpNwyV~uCJ9x1S5CdExr_-"
                "Z(vGh)b|G%I>?YOfwoTkVTN!KcZMg5PTEjV08U!4fIOE{Vx&I%<n~T%"
            )
        )
    ).decode("utf-8")
)
_REBALANCE_ACTIONS = json.loads(
    zlib.decompress(
        base64.b85decode(
            (
                "c-rk<O>bM*5&bV(a}iRq;&i9jFR>8CG9<Y~Y6wAqrYKOPi?q9<|GkRLkH<GNXU@#r`$)T9nWjj-"
                "`+aB5oH_I1KWBgY<@euy{r&7uAI{!g-QS-*F3$e`%Rm46uctSjUjFvW@BjGq-%qbUoV~fe8=n5kz4-"
                "S1Pd{J1z5em)=4^5H>UMp$Sc$jq-weaM(I4)I;q}wo>o>#I!`b5N$=g2;H@9!k7R%ko-`(B5e*fxW|BV-qkN;g9^ym835AWaYpR^"
                "qH?Zer6xPN$V>)YGAhd0l!c5h8y91g_&aC5VN>e77b?gQhe?*2L+l<S+<KR*rr!~4zXaqUnCK{RLc6Iv7Y!(u-"
                "&HwNILo3EVr@A>rCN3GdUxib0Fx5m$&&h^dJt6?{i2ak|_(;g~bfrouJ-XDj9yJ^NBYwC~Rdiwv{huxmh--"
                "$f=)zx$WCvABes;h_LZt`mT(EX<-%|IhNIg56U7`J@5x;LI4`sMu(%3<0)V)x?y_PsM)@(Id9-"
                "?e!6!)=Fax+a>Bnpgs|{mL^xzQ|4dzP)J1DuX7EpV47Z8m-"
                "N$VWxR8{&Zrz&|;IDv(3Ow55f*3NUS&c4!A}#>kbWH=3LmkGivAl9qUoK2PlKJ`_tr+$sX+6D}MOni@@)qj{@@wd~Lf5na^4mZQz"
                "DPAHBV~8D2g7{M~T(aD8+Am*-Jy-Gw>YGd1$SN4MwMm!_AS9@#2iJ_<eBjT1APg2jdH&IbI=%?B=^zZu!-p>NxMLe1uf-"
                ")5aMyzXN*La$+pv_TG21JAYPAW2u8w~55r7kjil!h8GHZG$qvs0poxc<&^64O9j=Fh<G{2RzO0YgoD;tp<3s1QKTF)O}8Rd2>IxU"
                "}{IMuEf0*J7ybPLYb?l$2!d2#)Y#lf8uqyO47$$@ZRvA;~r&+1FYlS8yYeHAh&PY)R4JH)2wSO?EZgFU&*>0lNTjdgNdh1qi=Crd"
                "(q2FiTyC)xHv!9%}if3^8#&jwQwgH5c`{}yFbV8tf`5wqV}7nq;V3(j0}zmvflT<Jq=`J?mh!ZE=`+UR=decPn?F=yHI;!2DLM)<"
                "lkx?;N3as^J=TDVA+f(YyX1>@5bj(&J~C-"
                "`j$!RkShgxc21Ivy)rABicy$LI~iTszEEbg_={HHrfPz0F16K*=VjoTCDKAz_V+R?K12H4I~gByZXf&g_4l33e|m>C@--"
                "{C8F3(5rcOtcEHq{x8yFihtsOZkQVW-g9lg)lZ=B23{M2GtB~Ne;uP@?#*vI}+Ls-CCK8+AoAVgA(VeJjNR0NF*fo1Sb;C{4)hRa"
                "^Gzl;f&IXpux+_P3w)X_eeaygTstQvQ%Q!H!isNCK@{j>aP>xMj8ldlWu2OBTu{^9Ow{q1mf_tVpFe`9hyR*xB%O)+RDUKg83k+`"
                "Fk3<C)pO4nMxpih<+0Txf&Y_v;qS7o~b{CwVw&~ZwIz#MqGZ-"
                "3yf2cO>1Uo)8F+z4&`z0np+n~Z|=t38=ZwTRqE_PSBB5W+*COlWEzf!(HH;TS1L7fG*$rL_p81Q~;xUT)H80a-"
                "2kwn`@d<jFZ4RCCI&re`1=r(oQIoz+m($ht+s)7OBt<cwEf_BtAHN|J$g*qiagq<OA3OPx_;nA}}pyj_Z)DKSO2q1t@9jR-i_#(v"
                "ZOiVpoilA5%d;8?fVdU-"
                "$&WdC%VJ>==gL>m|`tg#d=*8oh3_jmNQR&7&KdDJO85)}qJddq+>Cw&s&v0Nn5g|)!h<s;9~KJ?|5*)(PQIFm_rGQBFBuPiTUY0!"
                "}7IXk27ZL42<MaB`LAmH<_;)#qy%~Q)-"
                "Nuy8LQ(|#Q`xvknx>`%S7sxL61CE<@7%+@f21pI(*YZpUHeFiI+IU^?W*)IVoVj5Z$2vkyax+&J6j=#9Ve=go^VRln#2g^pOXrwy"
                "2S_#_Qq6c06h%EeIDqI>U$_;7MQf%j$A$%;!u(IGu?B@HnahIKU{hd0RtBWW??!k=%dZWd(dFp<AFpqI*h>6_%)l64zVwl4Q@J8$"
                "`5C6^lu5N<??-uGNmEfbS8F;q=sciKHM=22JQ(7!q}^Ia!uEEFap5u~$<}DFYwVg)g}M0g<`RO;`H1-}H_U-"
                "~bY%vaZ2Ed;B&e0lFP)qbs$2s|Q%M0)_s=a^`LxxYT^A(_aMy3e$Izsccn?eVd;ctc^366Ry9t8BRQeF#Nw92tnDUY}t~+u$^ByH"
                "`gLO;5j#)C3Jk53wd*=2os=TS)&l#&#K35TBYBRUeS{=Fn60|JQWO!S-"
                "29E}xl(*AodywuBJo9;8ju}29Nyv)X&)WYhz?WHail8B>%A^o#Lh;ns-CX5-"
                "1*Wr!1ka%FkZ1+UZzFvNoAuHIqGs(v3RtekiO~?LxrUJ>(gu!~J~(=77+F;6o5@623Lyopb&|_^(R2_+Va<#(BOqLWKxj{D01gc&"
                "idyjZl%2H3n<uc1(xVATG^*8%7iD<!G+hVaT0v2w;AzmP`PvH6m#wn>up?!O80T8Pdd$QIv;jxRg5SawKejNeqmA9`yW4jhEn02i"
                "fuWWjSDK_HR=4!K(4e;+YQSBv9f{p|?_P-cK?A+Ihz#_3n>Hhl^J6!nYva5!%p)z0T$}wQg}{(PTq4xK-HHJ1Ycr;5ERTucMV;ir"
                "xVH7&sn!KXkaG96$@&=Qy9B$KN%CR6{}y9dD22kZ*9;V3oT#)qZ$M##c!%;wKLKaqP;?D|nJ}zVT)A;slj)l)^OLb|T69%FP$`g("
                "U6$U3_U6ibQ|TgxTV%8Y@MMaVKv_xvXRbIr9SZnACgo5}F35ocbV-JjuUWy7>EnowyD3$QLd56KV~DfF@wT{=PmQ#LK`bR6W;?Xu"
                "3Q9hujDAwdJ%OcXVn@UbT{PN##@kdC6B{XFR%37#^CV=1hlCCrjE*6Um!c{Z`93xQ2&AoI_6>WdD{<_CJT|cx!L-"
                "3i^74MNt5HFA>7BdkyK|K&cyL`VIGx5)<~4%Ol;AnY$3aAtz~w&yK4N=Q1yw__@?oxZn(3ZaZ)$QoB$)x>6<1>T%6>#zO&=;s!Y^"
                "{6YXV;f=Y9$ZQ8rK}<*rxnk~>Ov&j}SY;gC$LPu5!OtuJ$Ea;>jz4=-P{!|IxCi-giZn{I@5Xj=&`vjG$ZFdHK=>D*8TSXT~Otqn"
                "HWppC*DaBFQFy+oyn*wA(pJaRy$W#_GoTc?rXWYT<m$zvw#EFda<Aiu=ygn<ail0!(;ya9^TF(a<QQegt?tEAp51`KCB7QuqGllC"
                "e`$LFrJ$&|E@B_P&bOaF9+0@k)I6_rzfGLJ6KvpKnpSH-`jK*s>uZKJyoG)oj8(8g0clafiwC|A~-"
                "zyK<hRBN$tT4CX;ar=_0C1LAll9O{!Z0)iAU!@m}vyC|#!q=Yx^WZq1Pb{kpl>yb<ql?FYSlAv=fDJ_%pM(-"
                "yO?*@immNoh1(h<HdW7H$<<p^i*9pR;6=rFnn6M)VH8g_*-"
                "kkcGaS*XOU1PaJV$H`15iIM$K4EIRz&7@fbPt>J&yV(uy8ud00v<piAV^jvsWzUSmV9UzT}rJNe{3pp@<*Xn&z>>);Wf*v>S5rp1"
                "eSiJK&PsZn4nE6e>3SR?oo-6jB}3&pP=5Ht3;n&uG-"
                "|;_#xDB7zmn2z=yRXO%fKD+u8Gg*KMoK$&eAwQFJp6s_kLBuU6pmJ?i$J`rvCI%<pU~*&>1bjQQY<IRJtKI6%oHx;&2hz$`JxX6`"
                "c+U2N(ghhmC_+87a^xNA(YrLiASz3mfC^bC0;nrM6A^ZKR%ws*wPinyLK<jB<&Zx*qRuC9P@maaTnS!ft40a4Eesj-"
                ">(BqvbNHaclq$MZWs^0;q7$VI#dnGwN;F@4ilnHTRdik~$~K;GzkrF?$LJO)9{G2Omeml_&-"
                "aec%tM9gx;=GZOx@}SQ%&S|<aqo`&VF||2T(m&UrW;1rDZW&NYo-"
                "G2Hkw77^m5G$xVnmaq@8P&oqafyqko@41t&KlpPElTGm+)C7y*KR`VmG6C>{6P^l;hZbuv;y5qMAwn5N3p>7cBA-"
                "U+|4Yf+I^;ISZP^fEgdr^NZm64!@$qk_3lCS$4ss)`7^kPnD{Jv`?SGn#pFYsY^g%IZC0RO~8=~GpZpdCA25?<paB1k@}U<9`@Qo"
                "lGz}})pn<&XqAjhDv7B;?W|xW0Eb6ws28w|MNvy(%A`hICm;E=T2l%^9*@3!F+;Vaj$qiQ6FXJDAu5%sf4RheX_p7uFjA{F_uOzx"
                "_ynbH@m-|VhKekoN)#Bi{}YZCAI|6I`-iKlSunSKjCn-CH=E9o!%r<Vz2=m~#~!RVb#unFl`R4qn2~)uS}Yv17hzrUQaOc$_Bgx$"
                "!PGF@3VepoL&n8+eVB=hx*xpCu1F$WubofsmIvENISxu5F;sw*m-"
                "yIi?zL1hslCdfR7V@LD}}0#jM&gLRe>x9sh|Rt8r$+z!OEmcL%J^!=?2AfH^9csb}kUUqr6{<o)|?u03SRM@noD!s*KzqIV-gV(W"
                "2!!#U!R-ty)-"
                "Bf<hPcy!?qdz?d<@(k4UvwCNS60>dybEU4duR#qKy$SE0o;;M~So(I<5_!P%N(rwyjd$77l$@pDBHzX=kCmT#SssZs~?6w63?7Jd"
                "9$_vjs<+A=P7pg!?=Ruw>wmRf#h~a4jgi=+^)`aR$Y)az01Mc8>Fy21kdx}ig^#VDuDY!{aFk3!am>}U7woQ`<gO3zcoC1HL4CIE"
                "J!6ht({B7+dr`NZA;OV=dJfeIKKMBrhuav@bxU}ZD3=<<mWDE%4@1XDys%KNGqwY$r0h)_4{LDRB=GsZ1z-"
                "}5swNRQgUsMGVSya+7DuanI0#e8#2?%j{^nh!F*f%ApifBZP%rV{j?tWi>Ra0)S7ZpUdP@ZLSP)Bpz;Y)*l@(m-"
                "NcYZYd;}ez2(>y^&J2Z=KWCg^_4eqNg&NZ)(NI(mrEwb6CtV)if#HG&klPp&*UmdcFaG5ir=Fp)=p*xQJl*^UMIy0pHdgrAZ-"
                "j1A$n3Z`C6|rw4(Wm3UmFTVOGO(oqn8bl(!8^74lw7ql+paqRDICl-"
                "2Gbh!+4MI%6w7A;;cW4=ELDd{{X&9nnSC2^m=zC~OGz;T>=ob)ibEiONoi-mp_F%sOF>b97wP3V3@j~4>%c4d#_(Ul*NWn4Oibma"
                "QncQ)dYrTr)4_sW2{9|AQ?1=5hzU}R^Q%OA#+{!m=3+By#lH$;;$se03dl?8aUY1-?ej-IaRK{NAtMAT+J|uZJnuj^t3W6%>QJ`K"
                "<6xqg&08!lCf%izwhuIIj_;zwC~TU24u&5S7APZXF!`TU$~n!FZ%0HBewxa@@!*mi9aQTY46BqJ(0t+vw{>v&QOvqe1^grFC&>!a"
                ")R7B3@f75kCNf|}7sdQXVgA%4EgNtbQaKN#!`fzs5_ATTCIQ_{Oe-"
                "l!SWxta?6$#2ffAbc=7ovOBQgZT*MT{X(RqkKMT(yc(S<mZ4Ua1JK3A#A_5o=y<;H_}ivo;+k>3oix+*|+A}YW@@}z2mzF>DUsz="
                "~Ut<vQh(dIw|n4H(HhS8NLYXk(2WW|`XA)i<+sLCX~$ZVTV?MkZj_ZzaRFvk>EYzfMoUs$>vU$qzDJSAC<lS5~0#R^-"
                "lL((0>x*a+JPy5mfp0lw$?^)K6;uoR<Kl~9?$x#VOVl-"
                "=UCxuf69SRU9)<cV^K8S_yKjP<5<`g5$urQ2bcGi|mrh3EJPDBX+#s9)O9R11?uzV^MR3s{`2g1gIZR@r`!`pPQm=~vD3cE8NVwR"
                "($yg_Wx&<bQ^G^;82?MXjK@0kpWBZ#~)7f7LK<UBSey*w~w9su8<cnTy&@yAp^NFl7v5m&(Jz-"
                "e+wE^tk&KbWICLLN`fa%#0^QMyG$G8b#}C^<`rCrPQ`Kn(cs{5OB2lw&BbAL;)m6^#+n3Zq?$6i9+XrFtqB=(5mD<V5BTe0o&GXA"
                "Ak-"
                "@S=_y4mY=NpJgf<2w`d9s)Egs%WpS;`@Rjb;=QwX%bjLwu}Vgib)C^lJp&`CluMBmja7oM$)zg$uYiR(G?OWg>(K=f(C3QM;Mvrm"
                "M`Xd+2~f}6tBVqHhp=FjCb$NYO|7%XHhXV%fz!dBM?(|Ox%|RTn}wYQQsJ#2aa@KtQLu=R5aTjl8tTzRbW}sq4>j5v=6eZS6N}F{"
                "i5c_cm^IQ)WyN(IC#U4Gw<$mBIh4B5h5FTu;pS1L{7nOt`wW*FV|0wYk;%P_x^19eAd8`m6Q<WREBaxOA%z!S7hxW9!IA7Xth<?F"
                "??Ap6Pg~<yxRMi<B1bI>WH!D+Z(jSU@z&9JhSE=p?Hor}Pd&;&<TK{?#p^^N-"
                "xbB4l%)ri2viXXK4V)$9WdQZ?s0`g$rmn^V{M_Vxsrc;Eilu%jCt6E-iVfY1-"
                "wYEs#24dK^<MJrV<u+dFi{A%08D5cGzhEc94F0W?bbJp$xFyh#*sq9o?0H2qKA#6c`dCwe5cL<9M~z{t3w=k*Yv^LRpy3Ch?mCfg"
                "c6s*&<vz3+^Hd_0)l#c6_!^LLSJbzac@<An)zPQM8m{zfoky<oyLM(nzM`+Sh6cb|G{1he}o1%m_?HgTk{^t4~0%HR_CL-"
                "47KI$XAXn3o+?n7i3RT377Mx7<xi7qyQ0I$C)DA9LX$Kl?NlZejIt;<U&G?joo*KW1U!mEjCzOCJLo4ts|jw>obiIkafdq>Kq3!R"
                "|Y97%i}@pSY(Ei-mO$=_Sy%I4GzqS4tHeTMF=M2r~76H;-"
                "_gls?AD*9iU5s;aF6Revw&Zq~@5%5~V{<!HQLd1?J##%8d%+7+Km~qtzzaohldrB)2&poPv!w22?^+o)A+~169u2iqU)g4;AR;iU"
                "hZF5CE8h;V8xGZI$%Ct98Jr4`pD?-"
                "X+19IccinWmRlH#K~4)vPI`*OHrjT!3#3A^Rj2PF|yi1t~(8LJEINe3Z>6mk}N95RM8=yvlf%y{nyvuhjLZ>azW}vn*8KH6ax2fw"
                "Rxe5j0pR7K7B(>TLsyoRUFBzQuIj=3X7@K4w-"
                "G|w*Sg+ZU2%mj%|bx?9PGo!*E{%2{P5`lrlOxHH~(kQIupm8@PBJh6~Lo<~B6V3+zZ4;7np_s-S{bb4|aacn)76VO-"
                ")k;Np}_B1g5`JBvVi(AWfj!Smkou?ptaC2gI`$&@c)qA`@%`fD&g*z&k4&}bbaLt3zW(K&T%$(x1*$#ZJ{QaDbrx)rL7vW0<k0j!"
                "9svd`Tk*Y6t?x3?9%UEov6x3o?l^Hd^9L@ysh0>w<xy$O+tteU-Bi~1YB0v3OkDl<uMGN>oV^b_#&tv{tQKrlXy4}7oQ5Ur<c!Qr"
                "tZ1Z04!R5eJHK;`h1Y6)3Ju^j$baX{T4qS|aC*70t{b?8=*lZRy_KWf!|F6oyiN7suA#a+03oL~vSmmNz0GDk1j?<L*IzGVYJyES"
                "VkqCj`-+JYQek15dm$y6YOmlDntT@Nr*^aM&3h7AH=o9WnP4L$ysrxmAlKURUfUI%fAz%WjJEBG-YeSD@OiScslWobFT^9)RtIX+"
                "zOm`7#|)KjZyGkSgpbEG>FkpL|n^u*GzvJ+$L0$SA3k<{SP?iVfOMoPB9Yyh6HgW}A}FZ(=KKvgT~=7mzOVSNfkN6Wh$G=?fEKY;"
                "cJR10!LN1EoD=8tIg9U0&*q+;rfR)*lCWa%j(0#ry5-"
                "~I?N%9mRV2~?ugt8v*#`p9PF4XHvap_x(+G26<sfow0IR|1*Sk<<}oUZTij>uDG`$d*>oFieyMs7C@-"
                "A*2aV`;{zJ``|E#q}T_EV}o371nR*`q8iB51(7=_*SS$X`--pNx%wiB)((MVpg2Y&LIc-cJa(z#$rK5WHot2TPwYChX>q=YC$6hV"
                "7QSR5ykpkzgm4!p@<y3))f%2Sd(6S*S|p*)Sin;^6nx6*$<*=mB9t6yk|A##p?Ifq(k2A>Dj-"
                "~F<2Y=SHRH<q+gmy_^{s)Z+x}*Vl^6zIG`;HCkwz;#5P%0wPOfpAkwZxQy9bfmd>~N=ku3Q=O_Vx>wH%uED$A3pX$cB);L4f^D?{"
                "Z7Vh2sLnhTr+zaz$wNm6w&ZFDzzITu{obW(^<lb>@y2B>wcNwMWH(wyKLX2w8at@t(HDmRVr{9~ATtA<I2Ak~q!T4b@r&cQ*foaQ"
                "!#PY!U`71Su(n!_(X1Nh{G3Ta;8vX+pfWdESapy)3urg5RQcCNfE6hnAzzQoCsu6qfq5+FH-"
                "j0#hvuw_Vqhha;dew9j+DI8HbJ6^d-3Q%lRgRNz8NlM6pg})MjX|h}-CYpI+^_Buw{YpSm69RoYF-"
                "L6@fU8v>qiZf9=F=!u!gZcVb$^lsn3Ny_3SMr0c|{h<L0V%hM^Z8^f_T@9!>(JTp2`<A?T0DPM-UGYD=8n^3Dv(;*&`v4Lou>Fz5"
                "xQC8ODkdt_v&;IqC{90P_`<#}dqnXt7P+#bjtUjEoe+@0c=jaxGSH7j5%}M6k?st?HnHT&|eF6(E(^xmcZQ0_I?eGF&6`&z4Rxz%"
                "6~^OX^v32uF(+4uqbm%ad`Dnu|X50ka(nYLa)a7<in1l#2*rQUgrQ#y2mIogA&Kpr_x(7w`+zuba2tAg3J4sz{mdMIq68twB0)k3"
                "xgYJBBbz_mYHY4!xj$G(W3Z;#?NI@Y1S)!UF>q%#cP$E0J^6+M;zcLw>d96US?NV$pxh_{kh32%AyVyEz?1n90!Sc;E;q3j85yGq"
                "L^Ul*9ETNz!Krr@h7uiY9(!)8SNil;MLo>)V$b&k{;dTA_ZR1C|tGBIOJ00%DYHhwZ&?lvfJb!AJuS9vkFZJAaHod!Fs3rhGv>RP"
                "T(;W)w<s%t<3uoR};(WwS}X9$jPD$tFY#gUWV9=T3d*5m_rJw7{kSr&2EE?o{%UZPET{-"
                "D{_W0ScDH*n)~GlZQ|%Ux8H;)?(9HSYZ>bDzXA3>Nr_e5-"
                "54K#06I?KIQG)umMgI9aOq>dBTWbDDlldQ5q1ie6|u<Vft`Ybt&IS%I{VZ**er{1g}oD5F?u(R_MkNYAYI^(-Bg;67>^-"
                "V&yC)5(hFnE3o-"
                "G6gTuT#^B$Z<Vet!3aJdSQWi%tU5Rz(03sM#k*MH{Umk)e=^HZ1!&Lwwq5+Q2WUIowItjNi^%Zj<L$ZV=NPDoQN<wQ4m^7%Az2^h"
                "O(tQY?b<)iUwS$NO5D;_kR(2{b_N5J<WE%i-"
                "w6t8?1LDipxBccbDCfqX$LK0Yf3pZENpOo1?4PkY;0|cq3N@ad9T!+ZZaU@3p}```&>3md6gGxR*!ra39V~3iCF+FPD)~heU=?s-"
                "!&GtR^1KB~eRP^VKWE4U(bHLmo@a^*L+12SLF8I_tVjN1MMOg_@|`|do!rs0O;m<=%X4CxSnDdO@+Y_|PnBjU_%hcHtaEIU0YFP+"
                "MLT>klJc{5$xxCbVU>13@~UKulyxklU+HRklRSNb2QMGK2BB~qA!aJGC}NGa&cMrzVJ;UUw?rlh3rGcZxhypda(aOZC(4A_1M-"
                "jm16*;D2m"
            )
        )
    ).decode("utf-8")
)
_PRICE_FLOOR = 1
_DEMAND_ALPHA = 0.25
_MARKET_PARAMS = {
    "WHEAT": (25, 10000, 400, "sqrt", 0.8, "log", 0.2),
    "CARROT": (35, 10000, 450, "log", 0.2, "sqrt", 0.7),
    "TOMATO": (60, 10000, 200, "linear", 0.4, "sqrt", 0.6),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.7, "linear", 1.6),
    "MELON": (250, 10000, 300, "log", 0.2, "sq", 3.6),
    "EGG": (50, 10000, 332, "linear", 0.4, "log", 0.2),
    "MILK": (160, 10000, 122, "sqrt", 0.6, "linear", 1.6),
    "WOOL": (200, 10000, 105, "log", 0.2, "sq", 3.2),
    "FERTILIZER": (100, 10000, 200, "linear", 0.4, "linear", 0.4),
}
_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
_WEED_STATE = {0: {}, 1: {}}
_WEED_REPLAY_STEPS = 8


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _regime(configuration):
    interval = int(_get(configuration, "townCenterSellInterval", 12) or 12)
    return "rebalance" if interval >= 24 else "legacy"


def _copy_action(action):
    action = copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }


def _seat(obs):
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _farm(obs, seat):
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _align_hands(action, obs):
    action = _copy_action(action)
    expected = len(_get(_farm(obs, _seat(obs)), "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(order or ["PASS"]) for order in hands[:expected]]
    return action


def _tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _trace_actor_action(actions, step, actor):
    trace = actions[min(max(int(step), 0), len(actions) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _weed_repair_action(obs, action, actions, step):
    action = _align_hands(action, obs)
    seat = _seat(obs)
    game = _WEED_STATE[seat]
    if step == 0 or step < game.get("last_step", -1):
        game = {"last_step": step, "active": {}}
        _WEED_STATE[seat] = game
    game["last_step"] = step
    farm = _farm(obs, seat)
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    active = game["active"]

    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - transaction["start"]
        if age == 1:
            unit_actions[index] = list(transaction["intended"])
        elif 2 <= age <= 1 + _WEED_REPLAY_STEPS:
            unit_actions[index] = _trace_actor_action(actions, step - 1, actor)
        else:
            active.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if intended[0] not in ("BUILD_PASTURE", "PLANT"):
            continue
        tile = _tile_at(farm, position)
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            continue
        active[actor] = {"start": step, "intended": list(intended)}
        unit_actions[index] = ["DIG"]

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _align_hands(action, obs)


def _shape(name, value):
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1.0 + value)
    raise ValueError(name)


def _market_price(item, inventory):
    base, equilibrium, scale, below_func, below_target, above_func, above_target = (
        _MARKET_PARAMS[item]
    )
    if inventory < equilibrium:
        amplitude = below_target * base / _shape(below_func, scale)
        price = base + amplitude * _shape(below_func, equilibrium - inventory)
    else:
        amplitude = above_target * base / _shape(above_func, scale)
        price = base - amplitude * _shape(above_func, inventory - equilibrium)
    return max(_PRICE_FLOOR, int(round(price)))


def _is_sell(order):
    return (
        isinstance(order, (list, tuple))
        and len(order) >= 3
        and order[0] == "SELL"
        and order[1] in _MARKET_PARAMS
    )


def _impact_score(obs, order):
    if not _is_sell(order):
        return float("-inf")
    item = str(order[1])
    try:
        quantity = max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0.0
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    current_quote = float(
        _get(prices, item, _market_price(item, current_inventory)) or 0
    )
    later_quote = float(_market_price(item, current_inventory + quantity))
    return float(quantity) * max(0.0, current_quote - later_quote)


def _demand_per_day(obs, configuration, item):
    town = _get(obs, "town", {}) or {}
    shops = list(_get(town, "unlocked_shops", []) or [])
    turns_per_day = int(_get(configuration, "turnsPerDay", 24) or 24)
    shop_interval = max(1, int(_get(configuration, "townShopSellInterval", 4) or 4))
    demand = 0.0
    for shop in shops:
        products = _SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += (turns_per_day / shop_interval) * (2 if len(products) == 1 else 1)
    regime = _regime(configuration)
    if item != "FERTILIZER":
        center_default = 24 if regime == "rebalance" else 12
        center_interval = max(
            1,
            int(
                _get(configuration, "townCenterSellInterval", center_default)
                or center_default
            ),
        )
        day = int(_get(obs, "day", int(_get(obs, "step", 0) or 0) // 24) or 0)
        multiplier = (
            1 if regime == "rebalance" else (4 if day >= 20 else 2 if day >= 10 else 1)
        )
        demand += (turns_per_day / center_interval) * multiplier
    return demand


def _order_score(obs, configuration, order):
    score = _impact_score(obs, order)
    if _regime(configuration) != "rebalance" or score <= 0 or not _is_sell(order):
        return score
    item = str(order[1])
    quantity = max(0, int(order[2]))
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    demand = max(0.25, _demand_per_day(obs, configuration, item))
    excess = max(0.0, current_inventory + quantity - 10000)
    urgency = min(1.0, (excess / demand) / 10.0)
    return score * (1.0 + _DEMAND_ALPHA * urgency)


def _rank_sell_slots(obs, action, configuration):
    action = _copy_action(action)
    market = list(action.get("market") or [])
    rows = [
        (_order_score(obs, configuration, order), -index, list(order))
        for index, order in enumerate(market)
        if _is_sell(order)
    ]
    if len(rows) < 2:
        return action
    rows.sort(reverse=True)
    ranked = iter(row[2] for row in rows)
    action["market"] = [next(ranked) if _is_sell(order) else order for order in market]
    return action


def agent(obs, configuration=None):
    try:
        actions = (
            _REBALANCE_ACTIONS
            if _regime(configuration) == "rebalance"
            else _LEGACY_ACTIONS
        )
        step = min(max(0, int(_get(obs, "step", 0) or 0)), len(actions) - 1)
        action = _weed_repair_action(obs, _copy_action(actions[step]), actions, step)
        return _align_hands(_rank_sell_slots(obs, action, configuration), obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }


def _kaggle_submission_entrypoint(obs, configuration=None):
    return agent(obs, configuration)
