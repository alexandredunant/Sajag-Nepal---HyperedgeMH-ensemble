1. Road impacts are joined to earthquake scenarios by row position, not keys.
Does it create issues or is it consistent if there is no row shuffling?

In the source tables, 693 of 2,310 road rows are positionally mismatched: nine of the
     30 event blocks are swapped.
What does it mean?


"Action: merge all three tables explicitly on DISTRICT and event, validate one-to-one matching, and regenerate every figure/
     result. This is especially important for the claim that the maxima occur in “the same earthquakes” (main_updt.tex:185)."
I would need to rebuild a new script and an other figure folder as output to check if the results are then different.

I would be surprised as I am normally pretty careful with merging. I would expect, if true, that it would seriously change the results and imply a lot of work ...

2. The remoteness calculation in the script does not match the manuscript and has an aggregation problem.
Fair enough, I realized that the weighting is indeed not consistent - can you check again where the numbers in the scripts were coming from?

- joins municipalities using names, although 22 municipality names occur in multiple districts;
  so same municipality names for different districts?
- creates 911 duplicated records through that join;
- sums municipality-level population percentages rather than weighting by population to obtain a district-level percentage.
What should I do? get one mobility figure per district first?

3. The “worst-case total” is not necessarily a possible earthquake outcome.
I think that using in the text "component-wise upper bound" is probably fairer

it could also be shown alongside the maximum of the within-event total


4. The scores add non-commensurate quantities.
I think it is fine as we are checking impact do I will go with this

we could add a sensitivity analysis to show ranking sensitivity to alternative weights 

5.Several statistical interpretations need tightening
Scenario exceedance frequency - probably fair and need amending the text and figures

I would be curious to compare with a coefficient of variation map - would need an additional map

"The results demonstrate a significant correlation between district remoteness and vulnerability to road damage" I think the figure 7 shows this

6. The Robinson comparison does not isolate the effect of adding landslides.
The idea here was more to show if the focus for the un protection service would change from distric to disctrict so it should be enough


7. Some operational conclusions exceed what was modelled.
"soften these to “could impede access” and “is consistent with an urban–rural divide,” or add road-network connectivity
     and urbanicity analyses." I think this is fair enough