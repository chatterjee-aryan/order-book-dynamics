# Prospective Features

We classify our features into three broad categories:

1. Order book related
2. Impulse related
3. Response related

## Order Book Related Features

These are features which primarily try to summarize key characterisitcs of the liquidity profile


1. **Touch Mid Price**

$$
M_t = \frac{B_0 + A_0}{2}
$$

where $B_0$ and $A_0$ are the best bid and ask prices.

2. **Microprice**

$$
MP_t =
\frac{
A_0 Q^B_0 + B_0 Q^A_0
}{
Q^B_0 + Q^A_0
}
$$

where $Q^B_0$ and $Q^A_0$ are the bid and ask quantities at the top of the book.

3. **Micro Displacement**

$$
D^{micro}_t = \frac{MP_t - M_t}{M_t}
$$

Positive values indicate that the top-of-book liquidity is leaning upward (bullish), while negative values indicate a downward lean.

4. **Book Bid**

The volume-weighted average bid price across the first $k$ levels:

$$
B^{book}_t =
\frac{
\sum_{i=0}^{k} B_i Q^B_i
}{
\sum_{i=0}^{k} Q^B_i
}
$$

5. **Book Ask**

The volume-weighted average ask price across the first $k$ levels:

$$
A^{book}_t =
\frac{
\sum_{i=0}^{k} A_i Q^A_i
}{
\sum_{i=0}^{k} Q^A_i
}
$$

6. **Book Mid Price**

$$
M^{book}_t =
\frac{
B^{book}_t + A^{book}_t
}{2}
$$

7. **Book to Mid Displacement**

$$
D^{book}_t =
\frac{
M^{book}_t - M_t
}{
M_t
}
$$

A negative value indicates that the center of displayed book liquidity lies below the current mid price, while a positive value indicates that it lies above the mid.

8. **Book to micro Displacement**

$$
D^{book}_t =
\frac{
M^{book}_t - MP_t
}{
MP_t
}
$$


9. **Touch Bid-Ask Spread**


$$
S_t =
\frac{
A_0 - B_0
}{
M_t
}
$$

10. **Book Bid-Ask Spread**

$$
S^{book}_t =
\frac{A^{book}_t - B^{book}_t }{M^{book}_t}
$$

Unlike the top-of-book spread, this measures the dispersion of liquidity deeper in the book.

11. **Bid-Side Passivity**

$$
P^B_t =
\frac{B_0 - B^{book}_t}{B_0}
$$

A larger value indicates that bid-side liquidity is positioned farther away from the touch.

12. **Ask-Side Passivity**

$$
P^A_t =
\frac{A^{book}_t - A_0}{A_0}
$$

A larger value indicates that ask-side liquidity is positioned farther away from the touch.

13. **Passivity Imbalance**

$$
PI_t =
P^A_t - P^B_t
$$

This measures the relative willingness of the two sides of the book to provide liquidity close to the touch.


14. **Top-of-Book Imbalance**



$$
I_0 =
\frac{
Q^B_0 - Q^A_0
}{
Q^B_0 + Q^A_0
}
$$

The normalized version lies between $-1$ and $1$.

15. **Level Imbalance**

For level $n$:

$$
I_n =
\frac{
Q^B_n - Q^A_n
}{
Q^B_n + Q^A_n
}
$$

This allows the directional structure of liquidity to be studied separately at different depths of the book.

16. **Total Depth Imbalance**

Across the first $k$ levels:

$$
I^{depth}_k =
\frac{
\sum_{i=0}^{k} Q^B_i
-
\sum_{i=0}^{k} Q^A_i
}{
\sum_{i=0}^{k} Q^B_i
+
\sum_{i=0}^{k} Q^A_i
}
$$

This measures the overall balance of displayed bid and ask liquidity.



17. **Bid Touch Concentration**

$$
C^B_k =
\frac{
Q^B_0
}{
\sum_{i=0}^{k} Q^B_i
}
$$

This measures how much of the total bid depth is concentrated at the touch.

18. **Ask Touch Concentration**

$$
C^A_k =
\frac{
Q^A_0
}{
\sum_{i=0}^{k} Q^A_i
}
$$

19. **Concentration Imbalance**

$$
CI_k =
C^B_k - C^A_k
$$

This captures whether one side has proportionally more liquidity concentrated close to the touch.

## Impulse Features

These are trade related features. Trades are aggressive or market orders and can be thought of as the most informational aspect of market microstructure study.

The only thing is that because these features tend to measure sums, they are dependent on the sampling rate or window size.

1. **Bids Hit**

This is the total number of bids which were hit, where the seller was the aggressor.

$$
n_B
$$

2. **Volume Hit**

$$
VH = \sum_{i=1}^{n_B} q_i
$$

where $q_i$ is the quantity for trade $i$.

3. **Offers Lifted**

Analogous to bids hit.

$$
n_A
$$

4. **Volume Lifted**

$$
VL = \sum_{i=1}^{n_A} q_i
$$

We do not include prices here because ask prices are greater than bid prices. Including prices would introduce an upward bias into this measure.

5. **Total Cost to Hit**

$$
CH = \sum_{i=1}^{n_B} q_i S_i
$$

where $S_i$ is the normalized bid-ask spread immediately before trade $i$.

6. **Total Cost to Lift**

$$
CL = \sum_{i=1}^{n_A} q_i S_i
$$

7. **Hitting Impulse**

$$
IH = \frac{CH}{CH + CL}
$$

8. **Lifting Impulse**

$$
IL = \frac{CL}{CH + CL}
$$

9. **Trade Volume Imbalance**

$$
VI = \frac{VL - VH}{VL + VH}
$$

Positive values indicate net aggressive buying pressure, while negative values indicate net aggressive selling pressure.

10. **Trade Count Imbalance**

$$
NI = \frac{n_A - n_B}{n_A + n_B}
$$

Positive values indicate that more offers were lifted than bids were hit.

11. **Impulse Imbalance**

$$
I = IL - IH
$$

Equivalently,

$$
I = \frac{CL - CH}{CL + CH}
$$

Positive values indicate that aggressive buying accounted for a greater share of spread-weighted trade pressure, while negative values indicate stronger aggressive selling pressure.

## Response Features

These features aim to summarize what liquidity providers have done in response to the impulse. Once again, these features depend on how one defines the window.

In order to construct these features, we perform a hypothetical replay in which we ignore all additions and cancellations to the order book during the window. We allow all trades to occur against the initial book, thereby constructing a counterfactual book.

This counterfactual book represents the state of the order book that would have existed if liquidity providers had taken no action during the window.

We denote $B_0$ and $A_0$ as the starting best bid and ask prices.

We then define the counterfactual bid:

$$
B_{cf}
$$

This is the best bid in the counterfactual scenario where the book is reconstructed without taking into account any additions or cancellations. In other words, it represents the best bid after all bearish trades have hit the initial order book during the window.

Similarly, we define the counterfactual ask:

$$
A_{cf}
$$

This represents the best ask after all bullish trades have lifted offers from the initial order book, again assuming no additions or cancellations by liquidity providers.

Finally, we denote the actual best bid and ask observed at the end of the window as $B$ and $A$.

1. **Bid-Side Change**

$$
\delta B =
\frac{
B - B_0
}{
B_0
}
$$

This measures the actual change in the best bid over the window.

2. **Ask-Side Change**

$$
\delta A =
\frac{
A - A_0
}{
A_0
}
$$

This measures the actual change in the best ask over the window.

3. **Bid-Side Counterfactual Change**

$$
\delta B_{cf} =
\frac{
B_{cf} - B_0
}{
B_0
}
$$

This measures how much the bid would have changed purely as a consequence of aggressive selling, assuming liquidity providers took no action.

4. **Ask-Side Counterfactual Change**

$$
\delta A_{cf} =
\frac{
A_{cf} - A_0
}{
A_0
}
$$

This measures how much the ask would have changed purely as a consequence of aggressive buying, assuming liquidity providers took no action.

5. **Bidding Response**

$$
BR =
\frac{
B - B_{cf}
}{
B_{cf}
}
$$

This measures how much stronger or weaker the actual bid is relative to the counterfactual bid.

A positive value indicates that the actual bid is above the counterfactual bid, implying a supportive bidding response from liquidity providers.

A negative value indicates that the actual bid is below the counterfactual bid, implying that bid-side liquidity has retreated more than would be explained by aggressive selling alone.

6. **Offering Response**

$$
OR =
\frac{
A_{cf} - A
}{
A_{cf}
}
$$

This measures how much stronger or weaker the actual offer is relative to the counterfactual offer.

A positive value indicates that the actual ask is below the counterfactual ask, implying that liquidity providers have stepped in and offered more aggressively.

A negative value indicates that the actual ask is above the counterfactual ask, implying that ask-side liquidity has retreated.

7. **Response Imbalance**

$$
RI =
BR - OR
$$

Positive values indicate that the liquidity-provider response is relatively more supportive on the bid side.

Negative values indicate that the liquidity-provider response is relatively stronger on the offer side.

8. **Ask Volume Response**

Let $VL$ denote the volume lifted during the window.

Under the counterfactual replay, this aggressive buying consumes liquidity from the initial ask-side book and results in the counterfactual best ask $A_{cf}$.

Using the final ask-side book, let $VA$ denote the displayed ask volume between the final best ask $A$ and the counterfactual ask $A_{cf}$.

We then define the ask volume response as:

$$
AVR =
\frac{
VA - VL
}{
VL
}
$$

The interpretation is:

- $AVR = 0$: liquidity providers have replaced the same amount of displayed ask volume as liquidity takers removed.
- $AVR > 0$: more ask-side liquidity is present in the depleted region than was originally removed.
- $-1 < AVR < 0$: some ask-side liquidity has been restored, but less than the amount removed.
- $AVR = -1$: no ask-side liquidity is present in the depleted region.

9. **Bid Volume Response**

Let $VH$ denote the volume that hit bids during the window.

Under the counterfactual replay, this aggressive selling consumes liquidity from the initial bid-side book and results in the counterfactual best bid $B_{cf}$.

Using the final bid-side book, let $VB$ denote the displayed bid volume between the counterfactual bid $B_{cf}$ and the final best bid $B$.

We then define the bid volume response as:

$$
BVR =
\frac{
VB - VH
}{
VH
}
$$

The interpretation is analogous to the ask volume response:

- $BVR = 0$: liquidity providers have replaced the same amount of displayed bid volume as liquidity takers removed.
- $BVR > 0$: more bid-side liquidity is present in the depleted region than was originally removed.
- $-1 < BVR < 0$: some bid-side liquidity has been restored, but less than the amount removed.
- $BVR = -1$: no bid-side liquidity is present in the depleted region.

10. **Volume Response Imbalance**

$$
VRI = BVR - AVR
$$

Positive values indicate that liquidity providers have responded more strongly on the bid side relative to the ask side.

Negative values indicate a relatively stronger ask-side liquidity response.