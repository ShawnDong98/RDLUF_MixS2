import argparse
import template

def merge_duf_mixs2_opt(parser):
    parser.add_argument('--spatial_branch', type=int, default=1, help='whether use spatial branch')
    parser.add_argument('--spectral_branch', type=int, default=1, help='whether use spectral branch')
    parser.add_argument('--spatial_interaction', type=int, default=1, help='whether use spatial interaction')
    parser.add_argument('--spectral_interaction', type=int, default=1, help='whether use spectral interaction')
    parser.add_argument('--stage_interaction', type=int, default=1, help='whether use stage interaction')
    parser.add_argument('--block_interaction', type=int, default=1, help='whether use block interaction')
    parser.add_argument('--in_dim', type=int, default=28, help='model\'s input dimension')
    parser.add_argument('--out_dim', type=int, default=28, help='model\'s output dimension')
    parser.add_argument('--dim', type=int, default=28, help='model\'s block dimension')
    parser.add_argument('--stage', type=int, default=7, help='number of model\'s stage')
    parser.add_argument('--DW_Expand', type=int, default=1, help='expand of depth-wise convolution')
    parser.add_argument('--ffn_name', type=str, default='Gated_Dconv_FeedForward', help='which feedforward function to use')
    parser.add_argument('--FFN_Expand', type=int, default=2.66, help='expand of FeedForward Network')
    parser.add_argument("--bias", type=bool, default=False, help="whether use bias")
    parser.add_argument('--LayerNorm_type', type=str, default="BiasFree", help="which LayerNorm type to use")
    parser.add_argument('--act_fn_name', type=str, default="gelu", help="which activation function to use")
    parser.add_argument('--body_share_params', type=int, default=1, help="whether stage body share parameters")

    return parser


