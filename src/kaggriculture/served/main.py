"""Derived from parent.py; improves final liquidation while preserving its core.

Keep the parent's joint worker assignment, crop and animal forecasts, wheat
preference, initial herd, and capacity-safe deposits. Broad harness tests showed
that changing the season-long crop preference hurt previously winning matchups.
Stop new fertilizer tasks on day 29: fertilizer pickup and application detours
competed with watering, harvesting and delivering already valuable crops when
only one watering remains. Sell unused fertilizer and spend those actions on
harvests instead; paired harness tests showed better closing banks and an
additional v54 win while preserving the tested stronger-matchup wins.
"""
import math

# The parent's wheat cycle remains the inexpensive fallback.
_DATA = {"WHEAT": (10, 4, 4), "CARROT": (20, 3, 3),
         "MELON": (80, 10, 6), "STRAWBERRY": (100, 16, 4),
         "TOMATO": (50, 11, 4)}
_RECIPES = {"bakery": {"EGG": 1, "WHEAT": 1},
 "pizza_shop": {"MILK": 1, "TOMATO": 1, "WHEAT": 1},
 "brunch_spot": {"EGG": 1, "WHEAT": 1, "STRAWBERRY": 1},
 "yarn_store": {"WOOL": 2}, "ice_cream_shop": {"STRAWBERRY": 1, "MILK": 1, "WHEAT": 1},
 "pet_cafe": {"CARROT": 2}, "smoothie_shop": {"STRAWBERRY": 1, "MILK": 1},
 "farmers_market": {"WHEAT": 1, "CARROT": 1, "TOMATO": 1, "STRAWBERRY": 1}}

def _price(crop, inventory):
    x = inventory - 10000
    if crop=="FERTILIZER": return max(1,100-0.2*x)
    if crop=="MILK": return max(1,160-256*x/122) if x>=0 else 160+96*math.sqrt(-x/122)
    if crop=="WOOL": return max(1,200-640*(x/105)**2) if x>=0 else 200+40*math.log1p(-x)/math.log(106)
    if crop=="EGG":
        u=abs(x)/332
        return max(1,50-10*math.log1p(x)/math.log(333)) if x>=0 else 50+20*(u+8*max(0,u-1)**2)
    if crop == "MELON":
        return max(1, 250 - 900*(x/300)**2) if x >= 0 else 250 + 50*math.log1p(-x)/math.log(301)
    if crop == "WHEAT":
        return max(1, 25-5*math.log1p(x)/math.log(401)) if x >= 0 else 25+20*math.sqrt(-x/400)
    if crop == "STRAWBERRY":
        return max(1,120-1.92*x) if x>=0 else 120+84*math.sqrt(-x/100)
    base, t, down, up = (35,450,24.5,35) if crop == "CARROT" else (60,200,36,24)
    u = abs(x)/t
    return max(1,base-down*math.sqrt(u)) if x>=0 else base+up*(u+8*max(0,u-1)**2)

def _move(pos, target):
    x,y = pos; tx,ty = target
    if x<tx: return ["EAST"]
    if x>tx: return ["WEST"]
    if y<ty: return ["SOUTH"]
    if y>ty: return ["NORTH"]
    return ["PASS"]

_ANIMALS={"GOOSE":(300,"EGG",4,2,"COOP"),"COW":(400,"MILK",8,1.5,"PASTURE"),"SHEEP":(500,"WOOL",6,4/3,"PASTURE")}

def _animal_work(tile, day, prices):
    """Value care after production, and feed against the actual final refresh."""
    animal=tile["animal"]
    _,product,delay,rate,_=_ANIMALS[animal]
    price=prices[product]
    first=tile["placed_day"]+delay
    interval={"GOOSE":1,"COW":2,"SHEEP":3}[animal]
    cap=4 if animal=="GOOSE" else 6
    production_tonight=day+1>=first and (day+1-first)%interval==0
    viable=day<29 and price*rate+prices["FERTILIZER"]>prices["WHEAT"]*1.2
    if day==28:
        pending=tile.get("pending_care_bonus",0)
        if tile["consecutive_unfed"]:
            output=min(cap,tile["yield_units"]+(1+pending if production_tonight else 0))
            viable=output*price+prices["FERTILIZER"]>prices["WHEAT"]
        else:
            extra=min(pending,max(0,cap-tile["yield_units"]-1)) if production_tonight else 0
            viable=extra*price>prices["WHEAT"]
    next_care_production=first+max(0,(day+2-first+interval-1)//interval)*interval
    care=(day<28 and next_care_production<=29 and price>15
          and (viable or tile["fed_today"])
          and (production_tonight or tile.get("pending_care_bonus",0)<cap-1))
    return viable and not tile["fed_today"], care and not tile["cared_today"]


def _crop_choices(o, spaces, demand, animal_count):
    """Allocate plots by marginal revenue at each actual harvest date."""
    day=o['day']; prices=o['market']['prices']; book=o['market']['inventory']
    supply={c:[0.0]*30 for c in _DATA}
    first={'WHEAT':2,'CARROT':2,'MELON':10,'TOMATO':8,'STRAWBERRY':10}
    for farm in o['farms']:
        for row in farm['tiles']:
            for t in row:
                if not isinstance(t,dict) or t.get('kind')!='PLANT': continue
                c=t['crop']; planted=t['planted_day']
                if c in ('TOMATO','STRAWBERRY'):
                    supply[c][day]+=t['yield_units']
                    interval=1 if c=='TOMATO' else 2
                    for k in range(4):
                        due=planted+first[c]+k*interval
                        if day<due<=29: supply[c][due]+=1.5
                else:
                    due=max(day,min(29,planted+_DATA[c][1]))
                    age=due-planted
                    if age>=first[c]:
                        amount=max(t['yield_units'],1+max(0,age-(6 if c=='MELON' else 2)+1))
                        if t.get('fertilized_until_day',-1)>=day: amount+=2 if c=='WHEAT' else 1
                        supply[c][due]+=min(6 if c in ('MELON','WHEAT') else 4,amount)
    expected={'WHEAT':3.75,'CARROT':2.25,'TOMATO':1.5,'STRAWBERRY':3.75,'MELON':0}
    consume={c:[0.0]*30 for c in _DATA}
    for c in _DATA:
        for due in range(day,30):
            consume[c][due]=demand[c]*(due-day)+0.65*sum(max(0,due-u)*expected[c] for u in range(3,25,3) if u>day)
    choices={}
    early_wheat=sum(isinstance(t,dict) and t.get('crop')=='WHEAT'
                    for row in o['farms'][o['player']]['tiles'] for t in row)
    spaces=sorted(spaces,key=lambda p:(min(abs(p[0]-x)+abs(p[1]-y) for x,y in ((4,4),(5,4),(4,5),(5,5))),p))
    for pos in spaces:
        best=None; bestscore=0; best_outputs=[]
        for c,(cost,duration,amount) in _DATA.items():
            if day+first[c]>29: continue
            if c in ('TOMATO','STRAWBERRY'):
                interval=1 if c=='TOMATO' else 2
                outputs=[(day+first[c]+k*interval,1.5) for k in range(4) if day+first[c]+k*interval<=29]
                # Account for the fertilizer that supports above-plain yield.
                cost+=prices['FERTILIZER']*len(outputs)/6
            else:
                due=min(29,day+duration);age=due-day
                amount=min(amount,1+max(0,age-(6 if c=='MELON' else 2)+1))
                bonus=2 if c=='WHEAT' else 1 if c=='CARROT' else 0
                if bonus and prices['FERTILIZER']*1.25<prices[c]*bonus:
                    amount=min(6 if c=='WHEAT' else 4,1+2*max(0,age-1))
                    cost+=prices['FERTILIZER']
                outputs=[(due,amount)]
            revenue=0.;added=0.
            for due,n in outputs:
                projected=book[c]+0.9*sum(supply[c][day:due+1])-consume[c][due]+added+n/2
                revenue+=n*_price(c,projected);added+=n
            score=(revenue-cost)/(outputs[-1][0]-day+0.5)
            if c=='MELON' and day<10: score*=1.2 if day>=3 else 1.38
            if c=='WHEAT': score*=1.6
            if score>bestscore: best=c;bestscore=score;best_outputs=outputs
        if day<4 and early_wheat<6:
            best='WHEAT';best_outputs=[(day+4,4)]
            early_wheat+=1
        if best:
            choices[pos]=best
            for due,n in best_outputs: supply[best][due]+=n
    return choices

def _animal_investment(obs, counts, demand):
    """Value our cash flow plus the price impact on visible competing output."""
    day=obs['day'];book=obs['market']['inventory'];horizon=29-day
    specs={'GOOSE':('EGG',4,1,4),'COW':('MILK',8,2,6),'SHEEP':('WOOL',6,3,6)}
    supply={p:[[0.0]*30 for _ in range(2)] for p in ('EGG','MILK','WOOL')}
    for seat,farm in enumerate(obs['farms']):
        for row in farm['tiles']:
            for t in row:
                if not isinstance(t,dict) or 'animal' not in t:continue
                p,first,interval,cap=specs[t['animal']]
                supply[p][seat][day]+=t.get('yield_units',0)
                due=t['placed_day']+first
                while due<=29:
                    if due>day:supply[p][seat][due]+=cap if due==t['placed_day']+first else interval+1
                    due+=interval
    best=None;bestscore=0.0;us=obs['player'];them=1-us
    for animal,(p,first,interval,cap) in specs.items():
        if day+first>27:continue
        revenue=0.0;added=0.0;running=0.0
        for due in range(day,30):
            own=supply[p][us][due];other=supply[p][them][due]
            running+=own+other
            rate={'EGG':1.5,'MILK':2.25,'WOOL':1.5}[p]
            consumption=demand[p]*(due-day)+sum(max(0,due-u)*rate for u in range(3,25,3) if u>day)*0.8
            base=book[p]+running*0.85-consumption
            n=(cap if due==day+first else interval+1) if due>=day+first and (due-day-first)%interval==0 else 0
            if n:revenue+=n*_price(p,base+added+n/2)
            # Our sales reduce competing banks as well as our own later quotes.
            impact=_price(p,base)-_price(p,base+added+n*0.5)
            revenue+=0.8*(other-own)*impact
            added+=n
        upkeep=(obs['market']['prices']['WHEAT']-min(35,obs['market']['prices']['FERTILIZER']*0.5))*horizon
        score=(revenue-upkeep-_ANIMALS[animal][0])/max(1,horizon)
        if score>bestscore:bestscore=score;best=animal
    return best


_TARGETS = {}

def _match_work(bids, tasks):
    """Assign workers jointly so a convenient task does not strand another unit."""
    if not bids:
        return []
    workers=sorted({b[2] for b in bids})
    plots=sorted({(tasks[b[3]][0],tasks[b[3]][1]) for b in bids})
    wi={w:i for i,w in enumerate(workers)}
    pi={p:i for i,p in enumerate(plots)}
    n=len(workers);m=len(plots)+n
    costs=[[0.0]*m for _ in workers]
    chosen={}
    for bid in bids:
        value,dist,w,j=bid
        row=wi[w];col=pi[tasks[j][0],tasks[j][1]]
        if value<costs[row][col]:
            costs[row][col]=value
            chosen[row,col]=bid
    u=[0.0]*(n+1);v=[0.0]*(m+1);owner=[0]*(m+1);way=[0]*(m+1)
    for row in range(1,n+1):
        owner[0]=row;j0=0
        best=[float('inf')]*(m+1);used=[False]*(m+1)
        while True:
            used[j0]=True;i0=owner[j0];delta=float('inf');j1=0
            for j in range(1,m+1):
                if used[j]:continue
                cur=costs[i0-1][j-1]-u[i0]-v[j]
                if cur<best[j]:best[j]=cur;way[j]=j0
                if best[j]<delta:delta=best[j];j1=j
            for j in range(m+1):
                if used[j]:u[owner[j]]+=delta;v[j]-=delta
                else:best[j]-=delta
            j0=j1
            if owner[j0]==0:break
        while True:
            j1=way[j0];owner[j0]=owner[j1];j0=j1
            if j0==0:break
    selected=[]
    for col in range(1,m+1):
        if owner[col] and (owner[col]-1,col-1) in chosen:
            selected.append(chosen[owner[col]-1,col-1])
    selected.sort()
    return selected+sorted(bids)

def agent(observation, configuration=None):
    o=observation; day=o["day"]; hour=o["hour"]
    farm=o["farms"][o["player"]]; tiles=farm["tiles"]
    private=o["private"]; seeds=dict(private["seeds"]); shed=private["shed"]
    if hour == 0: _TARGETS[o["player"]] = {}
    previous = _TARGETS.setdefault(o["player"], {})
    next_targets = {}
    positions=[farm["farmer"]]+farm["hands"]; invs=private["inventories"]
    half=len(tiles)//2; corners=[(half-1,half-1),(half,half-1),(half-1,half),(half,half)]
    market=[]; money=farm["money"]; prices=o["market"]["prices"]
    demand={c:1.0 for c in prices}
    for shop in o["town"]["unlocked_shops"]:
        for c,n in _RECIPES.get(str(shop).lower().replace(" ","_"),{}).items(): demand[c]+=6*n
    counts={c:0 for c in _DATA}; animals={c:0 for c in _ANIMALS}; own_animals=0
    for fi,f in enumerate(o["farms"]):
        for row in f["tiles"]:
            for t in row:
                if isinstance(t,dict):
                    if t.get("kind")=="PLANT": counts[t["crop"]]+=1
                    elif "animal" in t:
                        animals[t["animal"]]+=1
                        if fi==o["player"]: own_animals+=1
    animal_work={(x,y):_animal_work(t,day,prices) for y,row in enumerate(tiles) for x,t in enumerate(row) if isinstance(t,dict) and "animal" in t}
    fert_tiles={}
    for y,row in enumerate(tiles):
        for x,t in enumerate(row):
            if day==29 or not isinstance(t,dict) or t.get("kind")!="PLANT" or t["watered_today"] or t.get("fertilized_until_day",-1)>=day: continue
            c=t["crop"];age=day-t["planted_day"]
            bonus=2 if c=="WHEAT" and age==2 else 1 if c=="CARROT" and age==2 else 2 if c=="STRAWBERRY" and age in (9,13) and day<29 else 3 if c=="TOMATO" and age==7 else 1 if c=="TOMATO" and age==10 else 0
            if bonus and prices[c]*bonus>prices["FERTILIZER"]*1.25 and (day<29 or c in ("WHEAT","CARROT")): fert_tiles[x,y]=bonus
    food_need=sum(feed for feed,care in animal_work.values())
    carrying_food=sum(min(2,inv.get("WHEAT",0)) for inv in invs)
    # Carried supplies help, but keep feed accessible to other workers.
    carrying_food=min(carrying_food,max(0,food_need-3))
    for item,n in shed.items():
        if item in _ANIMALS: continue
        keep=max(0,food_need-carrying_food) if item=="WHEAT" and day<29 else len(fert_tiles) if item=="FERTILIZER" and (day<29 or hour<12) else 0
        if n>keep: market.append(["SELL",item,n-keep])
    if hour<4:
        area=sum(t!="LOCKED" for row in tiles for t in row)
        hires=min(12,max(7,(area+7)//8+(own_animals+3)//4))
        if day>=28: hires=min(hires,10)
        a,b=1,1
        for n in range(hires):
            cost=a; a,b=b,a+b
            if n<farm["hires_today"]: continue
            if len(market)>=10 or money<cost+50: break
            market.append(["HIRE"]);money-=cost
    if day<29 and food_need>shed.get("WHEAT",0)+carrying_food and len(market)<10:
        n=food_need-shed.get("WHEAT",0)-carrying_food
        market.append(["BUY_PRODUCT","WHEAT",n]); money-=n*prices["WHEAT"]
    held_fert=shed.get("FERTILIZER",0)+sum(inv.get("FERTILIZER",0) for inv in invs)
    if (day<29 or hour<12) and len(fert_tiles)>held_fert and money>500 and len(market)<9:
        n=min(len(fert_tiles)-held_fert,16,int((money-400)/max(1,prices["FERTILIZER"])))
        if n>0: market.append(["BUY_PRODUCT","FERTILIZER",n]);money-=n*prices["FERTILIZER"]
    land_cost={1:1000,2:2000,3:4000,4:1000000}[len(farm["unlocked_quadrants"])]
    if day<19 and day>2 and money>land_cost+(500 if len(farm["unlocked_quadrants"])==1 else 1000) and len(market)<9:
        market.append(["BUY_LAND"]); money-=land_cost
    existing=sum(shed.get(a,0)+sum(inv.get(a,0) for inv in invs) for a in _ANIMALS)
    # Compare animal investment returns with expected market saturation.
    best_animal=None; best_return=0
    if day==0 and own_animals+existing==0 and money>2000 and len(market)<8:
        market.extend([["BUY_ANIMAL","COW",2],["BUY_ANIMAL","SHEEP",2]])
        money-=1800
    elif day<15 and own_animals+existing<min(16,4*len(farm["unlocked_quadrants"])) and money>650:
        best_animal=_animal_investment(o,animals,demand)
        if best_animal and existing<2 and len(market)<9:
            cost=_ANIMALS[best_animal][0]
            if money>cost+400+own_animals*prices["WHEAT"]:
                market.append(["BUY_ANIMAL",best_animal,1]);money-=cost
    open_positions=[]
    for y,row in enumerate(tiles):
        for x,t in enumerate(row):
            if t!="LOCKED" and (t is None or (isinstance(t,dict) and t.get("kind") in ("WEED","COOP","PASTURE") and "animal" not in t)):
                open_positions.append((x,y))
    animal_available={a:shed.get(a,0)+sum(inv.get(a,0) for inv in invs) for a in _ANIMALS}
    animal_targets={}
    for a,n in (animal_available.items() if day<29 else []):
        for k in range(n):
            options=[p for p in open_positions if p not in animal_targets]
            if options:
                p=min(options,key=lambda q:min(abs(q[0]-cx)+abs(q[1]-cy) for cx,cy in corners))
                animal_targets[p]=a
    choices=_crop_choices(o,[pos for pos in open_positions if pos not in animal_targets],demand,own_animals)
    needs={c:0 for c in _DATA}
    for c in choices.values(): needs[c]+=1
    for c,n in sorted(needs.items(),key=lambda kv:-kv[1]):
        n=min(3,n)-seeds.get(c,0)
        if n>0 and money>350 and len(market)<10:
            n=min(n,max(0,int((money-300)/_DATA[c][0])))
            if n: market.append(["BUY_SEED",c,n]);money-=n*_DATA[c][0]
    tasks=[]
    for y,row in enumerate(tiles):
        for x,t in enumerate(row):
            if t=="LOCKED": continue
            action=None;value=0;required=None
            if isinstance(t,dict) and "animal" in t:
                product=_ANIMALS[t["animal"]][1]
                if t["yield_units"]:
                    tasks.append((x,y,["HARVEST"],t["yield_units"]*prices[product] if day==29 else min(750,40+t["yield_units"]*prices[product]*0.8),None))
                if animal_work[x,y][0]:
                    tasks.append((x,y,["FEED"],230 if t["consecutive_unfed"] else 135,"WHEAT"))
                if t["fertilizer_available"] and prices["FERTILIZER"]>8:
                    tasks.append((x,y,["COLLECT_FERTILIZER"],25+prices["FERTILIZER"]*0.6,None))
                if animal_work[x,y][1]:
                    tasks.append((x,y,["CARE"],min(105,25+prices[product]*0.5),None))
                continue
            elif (x,y) in animal_targets:
                a=animal_targets[x,y];structure=_ANIMALS[a][4]
                if t is None: action=["BUILD_"+structure];value=60
                elif t.get("kind")==structure: action=["PLACE",a];value=120;required=a
                else: action=["DIG"];value=60
            elif isinstance(t,dict) and t.get("kind")=="PLANT":
                c=t["crop"];age=day-t["planted_day"];finish=_DATA[c][1]
                ongoing=c in ("TOMATO","STRAWBERRY")
                cap = 6 if c in ("WHEAT", "MELON") else 4
                mature=(c=="WHEAT" and age>=3 and t["yield_units"]>=5) or age>=finish or (ongoing and t["yield_units"]>0) or (not ongoing and t["yield_units"]>=cap)
                if day==29 and age>=(2 if c in ("WHEAT","CARROT") else 10 if c=="MELON" else 8): mature=True
                home_distance=min(abs(x-cx)+abs(y-cy) for cx,cy in corners)
                terminal_harvest=day==29 and hour+home_distance>=21
                if mature and t["yield_units"]>0 and (t["watered_today"] or (c=="WHEAT" and age>=3 and t["yield_units"]>=5) or ongoing or (not ongoing and t["yield_units"]>=cap) or terminal_harvest):
                    action=["HARVEST"];value=t["yield_units"]*prices[c] if day==29 else max(150,min(700,t["yield_units"]*prices[c]*0.6))
                elif age>=finish and ongoing and not t["yield_units"]:
                    action=["DIG"];value=40 if day<=27 else 0
                elif not t["watered_today"] and (day<29 or (mature and not ongoing)):
                    productive=(c=="WHEAT" and 2<=age<=4) or (c=="CARROT" and 2<=age<=3) or (c=="MELON" and 6<=age<=10)
                    if productive or t["consecutive_unwatered"] or age==0 or (x,y) in fert_tiles or (ongoing and t.get("fertilized_until_day",-1)>=day):
                        action=["WATER"];value=110
            elif (x,y) in choices and hour<19:
                crop=choices[x,y]
                if seeds.get(crop,0)>0:
                    action=["PLANT",crop] if t is None else ["DIG"];value=65
            if action and action[0]=="WATER" and (x,y) in fert_tiles and (held_fert or any(m[0]=="BUY_PRODUCT" and m[1]=="FERTILIZER" for m in market)):
                if hour>=16: tasks.append((x,y,["WATER"],125 if t.get("consecutive_unwatered",0) else 85,None))
                action=["FERTILIZE"];value=135;required="FERTILIZER"
            if action and value>0: tasks.append((x,y,action,value,required))
    # Contiguous worker territories prevent centre chores from starving distant crops.
    ordered=[]
    for y,row in enumerate(tiles):
        xs=range(len(row)) if y%2==0 else range(len(row)-1,-1,-1)
        for x in xs:
            t=row[x]
            if t=="LOCKED": continue
            weight=1.4 if day<21 else 0.25
            if isinstance(t,dict):
                if "animal" in t: weight=6
                elif t.get("kind")=="PLANT": weight=1.4
            ordered.append(((x,y),weight))
    total=sum(w for p,w in ordered);cum=0;territory={}
    for p,w in ordered:
        territory[p]=min(len(positions)-1,int((cum+w/2)*len(positions)/total));cum+=w
    available=dict(shed)
    actions=[None]*len(positions)
    homes=[]; routes=[]
    for i,pos in enumerate(positions):
        inv=invs[i] if i<len(invs) else {}
        carrying=sum(max(0,n-(2 if c=="WHEAT" and day<29 and own_animals else 2 if c=="FERTILIZER" and fert_tiles else 0)) for c,n in inv.items() if c not in _ANIMALS)
        corner=min(corners,key=lambda q:abs(q[0]-pos[0])+abs(q[1]-pos[1]))
        home=abs(corner[0]-pos[0])+abs(corner[1]-pos[1])
        homes.append(home);routes.append(corner)
        sale_value=sum(n*prices.get(c,0) for c,n in inv.items() if c not in _ANIMALS and c not in ('WHEAT','FERTILIZER'))
        if (carrying or (day==29 and sum(inv.values()))) and (home==0 or carrying>=10 or sale_value>=650 or (day<29 and sum(sum(v for c,v in iv.items() if c not in _ANIMALS) for iv in invs)>80 and hour>=18-home) or (day==29 and hour>=22-home)):
            actions[i]=["DROP"] if home==0 else _move(pos,corner)
    bids=[]
    for i,pos in enumerate(positions):
        if actions[i] is not None: continue
        inv=invs[i] if i<len(invs) else {}
        for j,(x,y,action,value,required) in enumerate(tasks):
            if action[0]=="PLANT" and seeds.get(action[1],0)<=0:continue
            dist=abs(pos[0]-x)+abs(pos[1]-y)
            if required and not inv.get(required,0):
                if available.get(required,0)<=0:continue
                dist=homes[i]+min(abs(x-cx)+abs(y-cy) for cx,cy in corners)+1
            if hour+dist+(2 if action[0]=="PLANT" else 1 if action[0]=="FERTILIZE" else 0)>23:continue
            if day==29:
                return_distance=min(abs(x-cx)+abs(y-cy) for cx,cy in corners)
                if action[0] in ("HARVEST","COLLECT_FERTILIZER") and hour+dist+return_distance+1>22: continue
                if action[0]=="WATER" and hour+dist+return_distance+2>22: continue
                if action[0]=="FERTILIZE" and hour+dist+return_distance+3>22: continue
            val=value/(dist+1.8)
            if territory.get((x,y))!=i: val*=0.75
            if previous.get(i)==(x,y): val*=1.7
            if dist==0: val*=1.5
            bids.append((-val,dist,i,j))
    bids=_match_work(bids,tasks)
    reserved=set()
    for _,dist,i,j in bids:
        if actions[i] is not None:continue
        x,y,action,value,required=tasks[j]
        if (x,y) in reserved:continue
        if action[0]=="PLANT" and seeds.get(action[1],0)<=0:continue
        pos=positions[i];inv=invs[i] if i<len(invs) else {}
        if required and not inv.get(required,0):
            if available.get(required,0)<=0:continue
            if homes[i]==0:
                n=min(available[required],2 if required in ("WHEAT","FERTILIZER") else 1)
                available[required]-=n;actions[i]=["PICKUP",required,n]
            else:actions[i]=_move(pos,routes[i])
        elif list(pos)==[x,y]:
            actions[i]=action
            if action[0]=="PLANT":seeds[action[1]]-=1
        else:actions[i]=_move(pos,(x,y))
        reserved.add((x,y))
        next_targets[i]=(x,y)
    actions=[a if a is not None else ["PASS"] for a in actions]
    # Inspiration's terminal liquidation, with a shared capacity reservation.
    # Deposits precede sales, so PLACE preserves excess that DROP would erase.
    deposits={}
    room=max(0,100-sum(shed.values()))
    for i,action in enumerate(actions):
        if action[0]!="DROP": continue
        inv=invs[i]
        if sum(inv.values())<=room and not any(inv.get(a,0) for a in _ANIMALS):
            delivered=inv
        else:
            items=[item for item,n in inv.items() if n>0 and item in prices]
            if not room or not items:
                actions[i]=["PASS"]
                continue
            item=max(items,key=lambda c:prices[c])
            n=min(inv[item],room)
            actions[i]=["PLACE",item,n]
            delivered={item:n}
        for item,n in delivered.items():
            deposits[item]=deposits.get(item,0)+n
            room-=n
    sale_orders={order[1]:order for order in market if order[0]=="SELL"}
    for item,n in deposits.items():
        if item in _ANIMALS: continue
        if day<29 and (item=="WHEAT" or (item=="FERTILIZER" and fert_tiles)): continue
        if item in sale_orders: sale_orders[item][2]+=n
        else: market.insert(0,["SELL",item,shed.get(item,0)+n])
    _TARGETS[o["player"]] = next_targets
    return {"farmer":actions[0],"hands":actions[1:],"market":market[:10]}
