import sqlite3, statistics as st, sys, json
res={}
for r in (0,1):
    db=sqlite3.connect(f"e3a16c1-r{r}.sqlite"); S=dict(db.execute("select id,value from StringIds"))
    K=[(s,e,S[n]) for s,e,n in db.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    marks=[s for s,e,n in K if n=="vocab_parallel_embedding_kernel"]
    steps=[(marks[i],marks[i+1]) for i in range(len(marks)-1)]
    L=[(b-a)/1e6 for a,b in steps]; m=st.median(L); steps=[s for s,l in zip(steps,L) if l<1.5*m]
    hot=[];cold=[];nccl=[];waits=[];pubwait=[];hotL=[[] for _ in range(61)];coldL=[[] for _ in range(61)]
    for a,b in steps:
        ks=[k for k in K if a<=k[0]<b]
        mk=[k for k in ks if k[2]=="Marlin"]
        g=[mk[i:i+4] for i in range(0,len(mk),4)]
        hot.append(sum(x[1]-x[0] for G in g for x in G[:2])/1e6); cold.append(sum(x[1]-x[0] for G in g for x in G[2:])/1e6)
        for li,G in enumerate(g[:61]):
            hotL[li].append(sum(x[1]-x[0] for x in G[:2])/1e6); coldL[li].append(sum(x[1]-x[0] for x in G[2:])/1e6)
        nccl+= [(k[1]-k[0])/1e3 for k in ks if k[2].startswith("ncclDevKernel_AllReduce")]
        waits+=[(k[1]-k[0])/1e3 for k in ks if k[2]=="_wait"]
        # publish -> wait end = sidecar round trip as seen by GB300
        pubs=[k for k in ks if k[2]=="_publish"]; ws=[k for k in ks if k[2]=="_wait"]
        for p,w in zip(pubs,ws): pubwait.append((w[1]-p[1])/1e3)
    q=lambda v,p: sorted(v)[int(p*(len(v)-1))]
    res[r]=dict(steps=len(steps),hot_ms=st.median(hot),cold_ms=st.median(cold),
      nccl_us=dict(n_per_step=len(nccl)/len(steps),p10=q(nccl,.1),p50=q(nccl,.5),p90=q(nccl,.9),min=min(nccl),sum_ms=sum(nccl)/len(steps)/1e3),
      wait_us=dict(p10=q(waits,.1),p50=q(waits,.5),p90=q(waits,.9),max=max(waits),n_gt5us=sum(1 for w in waits if w>5)/len(steps)),
      publish_to_wait_end_us=dict(p10=q(pubwait,.1),p50=q(pubwait,.5),p90=q(pubwait,.9)),
      hotL=[round(st.median(x),4) for x in hotL],coldL=[round(st.median(x),4) for x in coldL])
    print(r,json.dumps({k:v for k,v in res[r].items() if k not in("hotL","coldL")}))
crit_cold=sum(max(a,b) for a,b in zip(res[0]["coldL"],res[1]["coldL"])); crit_hot=sum(max(a,b) for a,b in zip(res[0]["hotL"],res[1]["hotL"]))
crit_m=sum(max(a+c,b+d) for a,c,b,d in zip(res[0]["hotL"],res[0]["coldL"],res[1]["hotL"],res[1]["coldL"]))
print("crit cold",round(crit_cold,3),"crit hot",round(crit_hot,3),"crit marlin",round(crit_m,3))
print("coldL r0",res[0]["coldL"]); print("hotL r0",res[0]["hotL"])
json.dump(res,open("e3a16c1-detail.json","w"),indent=1)
